#![allow(dead_code)]
#![allow(unused_variables)]
#![feature(custom_derive)]
// #![feature(test)]

extern crate blas;
extern crate lapack;
extern crate rand;
// extern crate test;

use rand::{thread_rng, Rng};
use std::iter::*;
use std::cmp::*;

trait MatrixMap {
    fn matrix_map<Matrix, F>(&self, func : F) -> Matrix
        where F: FnMut(f64) -> f64;

}
#[derive(Debug, Clone, Map)]
pub struct Matrix {
    elements : Vec<f64>,
    row_size : usize,
    col_size : usize,
    transpose :  bool,
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
         (self.elements == other.elements)
       & (self.row_size == other.row_size)
       & (self.col_size == other.col_size)
       & (self.transpose  == other.transpose)
   }
}

impl Matrix{

    // create new Matrix
    fn new(e : Vec<f64>, r_size : usize, c_size : usize) -> Matrix{

        if (r_size * c_size) != e.len(){
            panic!("dimensions do not match length of vector.")
        }

        Matrix {
            elements : e,
            row_size : r_size,
            col_size : c_size,
            transpose : false,
        }
    }

    // creates matrix of zeros
    fn zeros(r_size : usize, c_size : usize) -> Matrix{
        Matrix {
            elements : vec![0.0;r_size*c_size],
            row_size : r_size,
            col_size : c_size,
            transpose : false,
        }

    }

    // creates matrix of random elements
    fn random(r_size : usize, c_size : usize) -> Matrix{
        Matrix {
            elements : rand::thread_rng()
            .gen_iter::<f64>()
            .take(r_size*c_size)
            .collect::<Vec<f64>>(),
            row_size : r_size,
            col_size : c_size,
            transpose : false,
        }
    }

    // map index in matrix to index in 1-d vector
    fn get_ind(&self, row :usize, col : usize) -> usize{
        if self.transpose == true{
            return (row-1)+ (col-1)*self.row_size
        }
        return (col-1) +(row-1)*self.col_size
    }

    // get an element from the matrix
    fn get_element(&self, row : usize, col : usize) -> f64{
        return self.elements[self.get_ind(row,col)]
    }


    // transpose matrix. We're not actually changing anything in memory.
    // we just flag the matrix as transpose to change pointer reference to elements.
    fn transpose(&self) -> Matrix {
        Matrix {
            elements : self.elements.to_owned(),
            row_size : self.col_size,
            col_size : self.row_size,
            transpose : match self.transpose { true => false, false => true}
        }
    }

    // get the diagonal of a matrix.
    fn diagonal (&self) -> Vec<f64>{
        let mut diag : Vec<f64> = Vec :: new();
        for elem in 1..min(self.row_size,self.col_size){
            diag.push(self.get_element(elem,elem));
        }
        return diag
    }


    fn tri (&self){
        unimplemented!()

    }

    fn triu (&self){
        unimplemented!()

    }


    fn tril (&self){
        unimplemented!()

    }


    // get a submatrix from a matrix.
    fn submatrix(&self, start : (usize,usize), dim : (usize,usize)) -> Matrix {
    unimplemented!();
    }
}

mod matrix_ops{
    use super::Matrix;
    use blas::*;
    pub fn multiply(a : &mut Matrix, b : &mut Matrix) -> Matrix{
            let m = a.row_size;
            let n = b.col_size;

            if a.col_size != b.row_size {
                panic!("number of columns in A does not match number of rows in B");
            }
            let k = a.col_size;
            let mut c = vec![0.0; m*n];
            dgemm(b'N', b'N', m, n, k, 1.0, &mut a.elements, m, &mut b.elements,k, 0.0,&mut c, m);
            Matrix {
                elements : c,
                row_size : m,
                col_size : n,
                transpose : false,
            }
    }


}

mod lp {
    use super::Matrix;
    use lapack::*;
    use std::cmp::*;
    use super::matrix_ops::*;
    // get the eigenvalues of a matrix.
    pub fn eigenvalues(a : &mut Matrix) -> Vec<f64>{
        let n = a.row_size;
        let mut w = vec![0.0; n];
        let mut work = vec![0.0; 4 * n];
        let lwork = 4 * n as isize;
        let mut info = 0;
        dsyev(b'V', b'U', n, &mut a.elements, n, &mut w, &mut work, lwork, &mut info);
        return w
    }

    pub fn lufact(a : &mut Matrix) -> (&mut Matrix, Vec<i32>) {
        let m = a.row_size;
        let n = a.col_size;
        let mut ipiv = vec![0; min(m,n)];
        let mut info = 0;
        dgetrf(m, n, &mut a.elements, m, &mut ipiv, &mut info);
        return (a, ipiv.to_owned())
    }

    pub fn lusolve(lufact : (&mut Matrix, Vec<i32>), b : &mut Matrix) ->  Matrix {
        let (a,mut ipiv) = lufact;
        let lda = a.row_size;
        let n = a.col_size;
        let ldb = b.row_size;
        let nrhs = b.col_size;
        let mut info = 0;
        dgetrs(b'N', n, nrhs, &mut a.elements, lda, &mut ipiv, &mut b.elements, ldb , &mut info);
        return b.to_owned()
    }

    pub fn qr(a : &mut Matrix) -> Matrix{
        let m = a.row_size;
        let n = a.col_size;
        let mut tau = vec![0.0; min(m,n)];
        let mut work = vec![0.0; 4*n];
        let lwork = 4*n as isize;
        let mut info = 0;
        dgeqrf(m, n, &mut a.elements, m, &mut tau,
        &mut work, lwork, &mut info);
        return a.to_owned()
    }

    pub fn singular_values(a : &mut Matrix) -> Vec<f64> {
            let mut at =  a.transpose();
            let mut adjoint_operator = multiply(a,&mut at);
            return eigenvalues(&mut adjoint_operator)
    }

    pub fn svd(a : &mut Matrix) -> (Vec<f64>,Vec<f64>,Vec<f64>) {
        let m = a.row_size;
        let n = a.col_size;

        let mut s = singular_values(a);
        let ldu = m*4;
        let mut u = vec![0.0; ldu*min(m,n)];

        let ldvt = n*4;
        let mut vt = vec![0.0;ldvt*n];

        let lwork = max(max(1,3*min(m,n)+min(m,n)),5*min(m,n)) +1 ;
        let mut work = vec![0.0; lwork];

        let mut info = 0;
        dgesvd(b'A', b'A',m,n,&mut a.elements,m,&mut s, &mut u,ldu, &mut vt, ldvt, &mut work, lwork as isize, &mut info);
        return (u,s, vt)
    }
}

#[cfg(test)]
mod tests{
    use super::Matrix;
    use super::lp::*;
    use super::matrix_ops::*;
    // use test::Bencher;
    #[test]
    fn test_zeros() {
        let row_size = 2;
        let column_size = 2;
        let mat = Matrix::zeros(row_size,column_size);
        assert_eq!(mat.elements, [0.0,0.0,0.0,0.0])
    }

    #[test]
    fn test_get_element() {
        let row_size = 2;
        let column_size = 2;
        let mat = Matrix :: new(vec![1.0,2.0,3.0,4.0],row_size,column_size);
        let element = mat.get_element(1,2);
        assert_eq!(2.0, element);
        let element = mat.transpose().get_element(1,2);
        assert_eq!(3.0, element)
    }

    #[test]
    fn test_transpose() {
        let row_size = 2;
        let column_size = 2;
        let mat = Matrix :: new(vec![1.0,2.0,3.0,4.0],row_size,column_size);
        let mat_t = mat.transpose().transpose();
        assert_eq!(mat_t,mat)
    }

    #[test]
    fn test_eigenvalues() {
        let mat = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3);
        let w = eigenvalues(&mut mat.to_owned());

        //tests will probably have to incorporate some semblence of error < eps...
        for (one, another) in w.iter().zip(&[2.0, 2.0, 5.0]) {
                assert!((one - another).abs() < 1e-14);
            }
    }

    #[test]
    fn test_singular_values() {
        let mat = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3);
        let w = singular_values(&mut mat.to_owned());
        println!("{:?}",w)
    }



    // #[bench]
    // fn bench_eig(ben : &mut Bencher){
    //     let mat = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3);
    //
    //     ben.iter( ||eigenvalues(&mut mat.to_owned()))
    //
    //
    // }
    #[test]
    fn test_lu_solve() {
        let mat = Matrix :: new(vec![-10.0,0.0,0.0,2.0],2,2);
        let mut k = mat.to_owned();
        let w = lufact(&mut k);
        let mut b =  Matrix :: new(vec![1.0,2.0],2,1);
        lusolve(w, &mut b);
    }

    #[test]
    fn test_multiply(){
        let mut a = Matrix ::new(vec![1.0,2.0],2,1);
        let mut b = Matrix ::new(vec![1.0,2.0],1,2);
        let C = multiply(&mut a,&mut b);
        println!("{:?}", C.elements)
    }
    // #[bench]
    // fn bench_lu_solve(ben : &mut Bencher){
    //     let mat = Matrix :: new(vec![-10.0,0.0,0.0,2.0],2,2);
    //     let mut k = mat.to_owned();
    //     let mut b =  Matrix :: new(vec![1.0,2.0],2,1);
    //     ben.iter( || lusolve(lufact(&mut k),&mut b))
    // }
}
