#![feature(test)]
extern crate test;
extern crate numbrs;
extern crate num;


use numbrs::{Matrix, Eig, Triangular};
use numbrs::scalars::*;
use numbrs::solvers::*;
use numbrs::operations::*;
use numbrs::factorizations::*;
use test::Bencher;
use num::traits::Float;

#[bench]
fn bench_eig(ben : &mut Bencher){
    let i = 250;
    let mut mat = Matrix ::random(i,i);
    ben.iter( ||eigenvalues(&mut mat,Eig :: Eigenvalues, Triangular :: Upper))
}

#[bench]
fn bench_dot(ben : &mut Bencher){
    let i = 500;
    let mut mat = Matrix ::random(i,i);
    let mut mat1= Matrix ::random(i,i);
    ben.iter( ||dot(&mut mat, &mut mat1))
}

#[bench]
fn bench_svd(ben : &mut Bencher){
    let i = 500;
    let mut mat = Matrix ::random(i,i);
    ben.iter( || svd(&mut mat))
}

    // #[bench]
    // fn bench_lu_solve(ben : &mut Bencher){
    //     let mut mat = Matrix ::random(2,2);
    //     let mut b =  Matrix :: random(2,1);
    //     ben.iter( || lusolve(lufact(&mut mat).ok().unwrap_or_else("MatrixError::ErrorGeneral"),&mut b))
    //
    //
    // }
    //
