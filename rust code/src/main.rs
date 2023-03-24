use clap::Parser;
use ndarray_rand::rand_distr::Distribution;
use petgraph::prelude as px;
use ndarray as nd;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr as ndr;
use num_complex::Complex32;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use std::collections::HashSet;
use std::time;

// fn iqp_sim_threaded(a: nd::ArrayView2<i8>, b: nd::ArrayViewMut1<i8>, depth: usize) -> Complex32 {
//     if depth == 0 {
//         iqp_sim(a, b)
//     } else {
//         let s1 = a.slice(nd::s![0, 1usize..]).sum() + b[0];
//         let c1 = Complex32::new(0.0, s1 as f32 * std::f32::consts::FRAC_PI_4).exp();

//         let mut b0 = b.to_owned();
//         let mut b0 = b0.view_mut();
//         b0.slice_mut(nd::s![1usize..])
//             .scaled_add(1, &a.slice(nd::s![1usize.., 0]));
//         let mut b1 = b;
//         b1.slice_mut(nd::s![1usize..])
//             .scaled_add(-1, &a.slice(nd::s![1usize.., 0]));
        
//         let (v0, v1) = rayon::join(
//             || iqp_sim_threaded(a.slice(nd::s![1.., 1..]), b0.slice_mut(nd::s![1..]), depth - 1), 
//             || iqp_sim_threaded(a.slice(nd::s![1.., 1..]), b1.slice_mut(nd::s![1..]), depth - 1)
//         );

//         v0 + c1 * v1
//     }
// }

fn iqp_sim(a: nd::ArrayView2<i8>, mut b: nd::ArrayViewMut1<i8>) -> Complex32 {
    
    if a.shape()[1] == 0 {
        return b.iter()
            .map(|&v| 1.0 + Complex32::new(0.0, v as f32 * std::f32::consts::FRAC_PI_4).exp())
            .product();
    }

    let s1 = a.slice(nd::s![0, 1..]).sum() + b[0];
    let c1 = Complex32::new(0.0, s1 as f32 * std::f32::consts::FRAC_PI_4).exp();
    let nb_connected : usize = a.slice(nd::s![1.., 0]).iter().filter(|&x| *x != 0).count();
    

    b.slice_mut(nd::s![1usize..])
        .scaled_add(1, &a.slice(nd::s![1.., 0]));

    let v0 = iqp_sim(a.slice(nd::s![1.., 1..]), b.slice_mut(nd::s![1..]));
    
    b.slice_mut(nd::s![1usize..])
        .scaled_add(-2, &a.slice(nd::s![1.., 0]));

    let v1 = iqp_sim(a.slice(nd::s![1.., 1..]), b.slice_mut(nd::s![1..]));

    b.slice_mut(nd::s![1usize..])
        .scaled_add(1, &a.slice(nd::s![1.., 0]));

    (v0 + c1 * v1) / ((1 << nb_connected) as f32).sqrt()
}

fn exact_mis<N: Clone, E: Clone>(mut graph: px::StableUnGraph<N, E>) -> HashSet<px::NodeIndex> {
    if graph.node_count() == 0 {
        return HashSet::new();
    } else if let Some(node) = graph.node_indices().find(|&ix| graph.neighbors(ix).count() <= 1) {
        let mut neighbours = graph.neighbors(node).detach();
        while let Some(neighbor) = neighbours.next_node(&graph) {
            graph.remove_node(neighbor);
        }
        graph.remove_node(node);

        let mut set = exact_mis(graph);
        set.insert(node);
        set
    } else if let Some(node) = graph.node_indices().find(|&ix| graph.neighbors(ix).count() >= 1) {
        let mut graph2 = graph.clone();
        graph.remove_node(node);
        let set1 = exact_mis(graph);

        let mut neighbours = graph2.neighbors(node).detach();
        while let Some(neighbor) = neighbours.next_node(&graph2) {
            graph2.remove_node(neighbor);
        }
        graph2.remove_node(node);
        let mut set2 = exact_mis(graph2);
        set2.insert(node);

        if set2.len() > set1.len() {
            set2
        } else {
            set1
        }
    } else {
        let components = petgraph::algo::tarjan_scc(&graph);
        for comp in &components {
            graph.remove_node(comp[0]);
        }
        exact_mis(graph)
    }
}

fn random_iqp(n: usize, gamma: Option<f32>) -> (usize, time::Duration) {
    let p = gamma.map(|gamma| gamma * (n as f32).ln() / (n as f32)).unwrap_or(1.0).min(1.0) * 0.75 ;
    let dist = rand::distributions::Bernoulli::new(p as f64).unwrap();
    let mut rng = rand::thread_rng();
    let mut a = nd::Array2::random((n, n), ndr::Uniform::new(5, 8));
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] *= dist.sample(&mut rng) as i8;
        }
    }
    let b = nd::Array1::random(n, ndr::Uniform::new(0, 8));
    for i in 0..n {
        a[(i, i)] = 0;
        for j in 0..i {
            a[(i, j)] = a[(j, i)]
        }
    }


    let mut graph = px::StableUnGraph::default();
    let nodes = (0..a.shape()[0]).map(|i| graph.add_node(i)).collect::<Vec<_>>();
    for (i, &ni) in nodes.iter().enumerate() {
        for (j, &nj) in nodes[..i].iter().enumerate() {
            if a[(i, j)] != 0 {
                graph.add_edge(ni, nj, ());
            }
        }
    }
    let mis = exact_mis(graph);

    let mut ordering = Vec::new();
    for i in 0..a.shape()[0] {
        if !mis.contains(&nodes[i]) {
            ordering.push(i);
        }
    }
    for &ni in &mis {
        let i = (0..a.shape()[0]).find(|&i| nodes[i] == ni).unwrap();
        ordering.push(i);
    }

    let mut a_perm = nd::Array2::zeros((a.shape()[0], a.shape()[1] - mis.len()));
    let mut b_perm = nd::Array1::zeros(b.shape()[0]);
    for i in 0..a_perm.shape()[0] {
        for j in 0..a_perm.shape()[1] {
            a_perm[(i, j)] = a[(ordering[i], ordering[j])];
        }

        b_perm[i] = b[ordering[i]];
    }

    let before = time::Instant::now();
    iqp_sim(a_perm.view(), b_perm.view_mut());
    (mis.len(), before.elapsed())
}

#[derive(Parser)]
struct Args {
    #[arg(short, long, help = "Value of gamma, if not specified, will use dense IQP")]
    gamma: Option<f32>,
    #[arg(help = "Minimum number of qubits")]
    nmin: usize,
    #[arg(help = "Maximum number of qubits")]
    nmax: usize,
    #[arg(help = "Number of repetitions per qubit")]
    reps: usize,
    #[arg(help = "Path of the output CSV file")]
    output: String
}

fn main() {
    let args = Args::parse();
    let mb = indicatif::MultiProgress::new();
    let kpb = mb.add(indicatif::ProgressBar::new(args.reps as u64)
        .with_style(indicatif::ProgressStyle::with_template("[{elapsed} elapsed, {eta} eta] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-")));
    kpb.tick();
    let runs = (0..args.reps).into_par_iter().map(|k| {
        let pb = mb.add(indicatif::ProgressBar::new(args.nmax as u64 - args.nmin as u64)
            .with_style(indicatif::ProgressStyle::with_template("[{elapsed} elapsed] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-")));
        let pbr = &pb;
        let runs = (args.nmin..=args.nmax).map(move |n| {
            pbr.set_message(format!("n = {n}, k = {k}"));
            let (mis, dur) = random_iqp(n, args.gamma);
            pbr.inc(1);
            (n, k, mis, dur.as_secs_f32())
        }).collect::<Vec<_>>();
        kpb.inc(1);
        pb.finish_and_clear();
        runs
    })
    .flatten_iter()
    .collect::<Vec<_>>();
    let mut writer = csv::Writer::from_path(args.output).unwrap();
    for run in runs {
        writer.serialize(run).unwrap();
    }
    kpb.finish_and_clear();
}
