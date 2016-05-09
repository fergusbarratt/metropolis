extern crate gnuplot as gn;
extern crate nalgebra as na;
extern crate rand;
extern crate itertools;

use na::{DMat};
use rand::{Rng, thread_rng};
use rand::distributions::{Sample, Range};
use std::vec::Vec;
use gn::{Figure, Caption, Color};

#[derive(Debug, Clone)]
struct SpinChain {
    len:usize,
    spins:Vec<f64>,

    coupling:DMat<f64>,
    field:f64,
    beta:f64
}

impl SpinChain {
    fn new(len: usize, coupling:DMat<f64>, field:f64, beta:f64) -> SpinChain {
        SpinChain {len:len, 
                   coupling:coupling, 
                   field:field, 
                   beta:beta,
                   spins:rand_spin_chain(len, &mut thread_rng(), &mut Range::new(0.0, 2.0))}
    }

    fn set_spins(&self, new_spins:&Vec<f64>) -> SpinChain {
        SpinChain {len:self.len.clone(),
                   coupling:self.coupling.clone(),
                   field:self.field.clone(),
                   beta:self.beta.clone(),
                   spins:new_spins.clone()}
    }

    fn len(&self) -> usize {
        self.len
    }

    fn energy(&self) -> f64 {
        self.beta * mat_sum(hadamard(&self.coupling, 
                                &outer(&self.spins, &self.spins))) +
        self.field * spin_sum(&self.spins)
    }
    
    fn mean_mag(self) -> f64 {
        1.0 / (self.len() as f64) * spin_sum(&self.spins)
    }
}

fn rand_spin_chain<R, D>(spins:usize, mut rng: &mut R, dist: &mut D) -> Vec<f64> 
    where R: Rng, D: Sample<f64>
{
    let blank = vec![0; spins];
    blank.iter()
            .map(|_| dist.sample(&mut rng) as i64 as f64) 
            .collect::<Vec<_>>()
}

fn gen_rand_ind(chain:&SpinChain) -> usize 
{
    thread_rng().gen_range(0, chain.len())
}

fn spin_flip(chain:&SpinChain, spin_ind:usize) -> SpinChain 
{
    let mut ret = vec![];
    ret.reserve(chain.len());
    for (ind, spin) in chain.spins.iter().enumerate() {
        if ind==spin_ind {
            ret.push(1.0-*spin);
        }
        else {
            ret.push(*spin);
        }
    }
    chain.set_spins(&ret) 
}

fn update_spin_chain(chain:&SpinChain) -> SpinChain
{
    let rand_ind = gen_rand_ind(chain);
    let prob = thread_rng().gen_range(0.0, 1.0);
    let del_e = spin_flip(chain, rand_ind).energy() - chain.energy();
    if del_e <= 0.0 {
        spin_flip(chain, rand_ind)
    }
    else if prob <= (-chain.beta * del_e).exp() {
       spin_flip(chain, rand_ind) 
    }
    else {
        chain.clone()
    }
}

fn spin_sum(vec:&Vec<f64>) -> f64 
{
    vec.iter().fold(0.0, |sum, x| sum + x) 
}

fn outer(vec_1:&Vec<f64>, vec_2:&Vec<f64>) -> DMat<f64> 
{
   DMat::from_fn(vec_1.len(), vec_2.len(), |i, j| vec_1[i] * vec_2[j])
}

fn hadamard(mat_1:&DMat<f64>, mat_2:&DMat<f64>) -> DMat<f64>
{
    let vec_1 = mat_1.clone().into_vec();
    let vec_2 = mat_2.clone().into_vec();
    let zip_ret_vec = vec_1.iter().zip(vec_2);
    let mut ret_vec: Vec<f64> = vec![];
    ret_vec.reserve(2*zip_ret_vec.len());
    for tup in zip_ret_vec {
        ret_vec.push(tup.0*tup.1);
    }
    DMat::from_col_vec(mat_1.nrows(), mat_2.ncols(), &ret_vec) 
}

fn mat_sum(mat:DMat<f64>) -> f64 
{
    mat.into_vec().iter().fold(0.0, |sum, x| sum + x)
}

fn main() 
{
    let length = 100;
    let field = 10.0;
    let beta = 0.00001;
    let n_iters = 100;
    let coupling: DMat<f64> = DMat::from_fn(length, length, |i, j| if i==j {0.0} else {1.0});
    let times = itertools::linspace(0.0, 10.0, n_iters).collect::<Vec<f64>>();
    let mut mags = vec![];
    mags.reserve(n_iters);

    let mut sys = SpinChain::new(length, coupling, field, beta);
    for _ in times.clone() {
        mags.push(sys.clone().mean_mag());
        sys = update_spin_chain(&sys);
    }
    
    let mut fg = Figure::new();
    fg.axes2d()
        .lines(&times.clone(), &mags, &[Caption("A line"), Color("blue")]);
    fg.show();
}
