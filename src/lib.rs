#![allow(dead_code)]
#![allow(unused_variables)]
#![feature(custom_derive)]
#![feature(test)]

extern crate blas;
extern crate lapack;
extern crate rand;
extern crate zipWith;
extern crate num;
extern crate test;

pub mod matrixerror;

use matrixerror::MatrixError;
use rand::{thread_rng, Rng, Rand};
use std::cmp::*;
use std::ops::{Add, Sub, Mul, Div};
use num::traits::{Num, Zero, One};
use zipWith::IntoZipWith;



#[derive(Debug, Clone)]
pub struct Matrix <T : Num + Clone + Rand> {
    elements : Vec<T>,
    row_size : usize,
    col_size : usize,
    transpose :  bool,
}


impl<'a, 'b, T : Num + Clone + Rand> Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Result<Matrix<T>, MatrixError>;

    fn add(self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        match self.col_size == other.col_size {
            true =>
            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x+y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : false}),
            false => return Err(MatrixError::MismatchedDimensions)
        }

    }
}

impl<'a, 'b, T : Num + Clone + Rand> Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Result<Matrix<T>, MatrixError>;

    fn sub(self, other: &'b Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        match self.col_size == other.col_size {
            true =>
            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x-y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : false}),
            false => return Err(MatrixError::MismatchedDimensions)
        }
    }
}

impl<'a, 'b, T : Num + Clone + Rand> Div<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Result<Matrix<T>, MatrixError>;

    fn div(self, other: &'b Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        match self.col_size == other.col_size {
            true =>
            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x/y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : false}),
            false => return Err(MatrixError::MismatchedDimensions)
        }
    }
}



impl<'a, 'b, T : Num + Clone + Rand> Mul <&'b Matrix<T>> for &'a Matrix<T> {
    type Output =Result<Matrix<T>, MatrixError>;

    fn mul(self, other: &'b Matrix<T>) ->Result<Matrix<T>, MatrixError>{
        match self.col_size == other.col_size {
            true =>
            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x*y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : false}),
            false => return Err(MatrixError::MismatchedDimensions)
        }
    }
}

impl <T : Num + Clone + Rand> PartialEq for Matrix<T>{
    fn eq(&self, other: &Matrix<T>) -> bool {
         (self.elements == other.elements)
       & (self.row_size == other.row_size)
       & (self.col_size == other.col_size)
       & (self.transpose  == other.transpose)
   }
}


impl <T:Num + Clone + Rand> Matrix <T>{

    // create new Matrix
    fn new(e : Vec<T>, r_size : usize, c_size : usize) -> Result<Matrix<T>, MatrixError>{
        if r_size * c_size != e.len(){
            return Err(MatrixError :: MalformedMatrix)
        }
        Ok(Matrix {
            elements : e,
            row_size : r_size,
            col_size : c_size,
            transpose : false,
        })
    }

    // creates matrix of zeros
    fn zeros (r_size : usize, c_size : usize) -> Matrix<T>{
        Matrix {
            elements : vec![Zero::zero();r_size*c_size],
            row_size : r_size,
            col_size : c_size,
            transpose : false,
        }
    }

    // creates matrix of random elements
    fn random(r_size : usize, c_size : usize) -> Matrix<T>{
        let e = rand::thread_rng()
        .gen_iter::<T>()
        .take(r_size*c_size)
        .collect::<Vec<T>>();

        Matrix {
            elements : e,
            row_size : r_size,
            col_size : c_size,
            transpose : false,
        }
    }

    fn diag_mat (a : Vec<T>) -> Matrix<T> {
        let mut mat = Matrix :: zeros(a.len(),a.len());
        for i in 1..a.len()+1{
            let e = &a[i-1];
            mat.replace(i, i, e.to_owned());
        }
        return mat
    }

    fn identity(row_size : usize) -> Matrix<T> {
          Matrix :: diag_mat(vec![One::one(); row_size])
    }


    fn replace(&mut self,row: usize, col:usize, value : T) -> () {
        let ind = self.get_ind(row,col);
        self.elements[ind] = value;
    }

    // map index in matrix to index in 1-d vector
    fn get_ind(&self, row :usize, col : usize) -> usize{
        if self.transpose == true{
            return (row-1)+ (col-1)*self.row_size
        }
        return (col-1) +(row-1)*self.col_size
    }

    // get an element from the matrix
    fn get_element(&self, row : usize, col : usize) -> T{
        let elem = &self.elements[self.get_ind(row,col)];
        return elem.to_owned()
    }


    // transpose matrix. We're not actually changing anything in memory.
    // we just flag the matrix as transpose to change pointer reference to elements.
    fn transpose(&self) -> Matrix<T> {
        Matrix {
            elements : self.elements.clone(),
            row_size : self.col_size,
            col_size : self.row_size,
            transpose : match self.transpose { true => false, false => true}
        }
    }

    // get the diagonal of a matrix.
    fn diagonal (&self) -> Vec<T>{
        let mut diag : Vec<T> = Vec :: new();
        for elem in 1..min(self.row_size,self.col_size){
            diag.push(self.get_element(elem,elem));
        }
        return diag
    }


    fn tri (row_size:usize, col_size : usize, k : usize, upper_or_lower : u8) -> Matrix<T>{
        let mut mat : Matrix<T> = Matrix :: zeros(row_size, col_size);
        for i in 1..row_size+1{
            for j in 1..col_size+1{
                match upper_or_lower{
                    b'U' =>{
                        match i <= j + k{
                            true => mat.replace(i,j,One::one()),
                            false => continue
                        }
                    }
                    b'L' => {
                        match i >= j + k{
                            true => mat.replace(i,j,One::one()),
                            false => continue
                        }
                    }
                    _ => {
                        panic!("upper_or_lower parameter must be U or L");
                    }
                }
            }
        }
        mat
    }



    // get a submatrix from a matrix.
    fn submatrix(&self, start : (usize,usize), dim : (usize,usize)) -> Matrix<T> {
    unimplemented!();
    }
}

mod operations{
    use super::Matrix;
    use blas::*;
    use num::traits::Num;
    use rand :: Rand;
    use matrixerror::MatrixError;

    pub fn dot (a : &mut Matrix<f64>, b : &mut Matrix<f64>) -> Result<Matrix<f64>, MatrixError>{
            let m = a.row_size;
            let n = b.col_size;

            match a.col_size == b.col_size {
                true =>{
                let k = a.col_size;
                let mut c = vec![0.0; m*n];
                dgemm(b'N', b'N', m, n, k, 1.0, &mut a.elements, m, &mut b.elements,k, 0.0,&mut c, m);
                Ok(Matrix {
                    elements : c,
                    row_size : m,
                    col_size : n,
                    transpose : false,
                })
            }
                false => return Err(MatrixError::MismatchedDimensions)
            }

    }


    pub fn matrix_map <T: Num + Clone + Rand> (func : &Fn(&T) -> T, a : &mut Matrix<T>) -> Result<Matrix<T>, MatrixError>{
          Ok(Matrix {
               elements: a.elements.iter().map(func).collect(),
               row_size : a.row_size,
               col_size : a.col_size,
               transpose : false,
           })
   }


    pub fn triu<T : Num + Clone + Rand>(a: &mut Matrix<T>, k: usize ) -> Result<Matrix<T>, MatrixError>{
        let tri_mat = Matrix :: tri(a.row_size, a.col_size, k, b'U');
        &tri_mat *&a
    }

    pub fn tril<T : Num + Clone + Rand>(a: &mut Matrix<T>, k: usize ) -> Result<Matrix<T>, MatrixError>{
        let tri_mat = Matrix :: tri(a.row_size, a.col_size, k, b'L');
        &tri_mat *&a
    }


}


mod lp {
    use super::Matrix;
    use lapack::*;
    use std::cmp::{min,max};
    use operations::*;
    use matrixerror::MatrixError;
    // get the eigenvalues of a matrix.
    pub fn eigenvalues(a : &mut Matrix<f64>) -> Matrix<f64>{
        let n = a.row_size;
        let mut w = vec![0.0; n];
        let mut work = vec![0.0; 4 * n];
        let lwork = 4 * n as isize;
        let mut info = 0;
        dsyev(b'V', b'U', n, &mut a.elements, n, &mut w, &mut work, lwork, &mut info);
        Matrix {
            elements : w.to_owned(),
            row_size : w.len(),
            col_size : 1,
            transpose : false,
        }
    }



    pub fn lufact(a : &mut Matrix<f64>) -> (&mut Matrix<f64>, Vec<i32>) {
        let m = a.row_size;
        let n = a.col_size;
        let mut ipiv = vec![0; min(m,n)];
        let mut info = 0;
        dgetrf(m, n, &mut a.elements, m, &mut ipiv, &mut info);
        (a, ipiv)
    }

    pub fn lusolve(lufact : (&mut Matrix<f64>, Vec<i32>), b : &mut Matrix<f64>) ->  Matrix<f64> {
        let (a,mut ipiv) = lufact;
        let lda = a.row_size;
        let n = a.col_size;
        let ldb = b.row_size;
        let nrhs = b.col_size;
        let mut info = 0;
        dgetrs(b'N', n, nrhs, &mut a.elements, lda, &mut ipiv, &mut b.elements, ldb , &mut info);
        Matrix {
            elements : b.elements.to_owned(),
            row_size : ldb,
            col_size : nrhs,
            transpose : false
        }
    }

    pub fn qr(a : &mut Matrix<f64>) -> Matrix<f64>{
        let m = a.row_size;
        let n = a.col_size;
        let mut tau = vec![0.0; min(m,n)];
        let mut work = vec![0.0; 4*n];
        let lwork = 4*n as isize;
        let mut info = 0;
        dgeqrf(m, n, &mut a.elements, m, &mut tau,
        &mut work, lwork, &mut info);
        Matrix {
            elements : a.elements.to_owned(),
            row_size : m,
            col_size : n,
            transpose : false
        }
    }

    pub fn singular_values(a : &mut Matrix<f64>) -> Result<Matrix<f64>, MatrixError> {
            let mut at =  a.transpose();
            let adjoint_operator = dot(a,&mut at);
             let mut e = eigenvalues(&mut adjoint_operator.unwrap());
             match matrix_map(&|x : &f64| x.sqrt(), &mut e) {
                    Ok(mat) => Ok(mat),
                    Err(mat) => Err(mat),
             }

    }

    pub fn svd(a : &mut Matrix<f64>) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>) {
        let m = a.row_size;
        let n = a.col_size;

        let s = singular_values(a);
        let ldu = m;
        let mut u = vec![0.0; ldu*min(m,n)];

        let ldvt = n;
        let mut vt = vec![0.0;ldvt*n];

        let lwork = max(max(1,3*min(m,n)+min(m,n)),5*min(m,n)) +1 ;
        let mut work = vec![0.0; lwork];

        let mut info = 0;
        let mut s_elem = s.unwrap().elements;
        dgesvd(b'A', b'A',m,n,&mut a.elements,m,&mut s_elem, &mut u,ldu, &mut vt, ldvt, &mut work, lwork as isize, &mut info);

        (
            Matrix {
                elements : u,
                row_size : n,
                col_size : ldu,
                transpose : false,
            },
            Matrix :: diag_mat(s_elem)
            ,
            Matrix {
                elements : vt,
                row_size : n,
                col_size : ldvt,
                transpose : true,
            }
        )
    }
}

#[cfg(test)]
mod tests{
    use super::Matrix;
    use super::lp::*;
    use super::operations::*;
    use test::Bencher;
    #[test]
    fn test_zeros() {
        let row_size = 2;
        let column_size = 2;
        let mat : Matrix <f64> = Matrix::zeros(row_size,column_size);
        assert_eq!(mat.elements, [0.0,0.0,0.0,0.0])
    }

    #[test]
    fn test_get_element() {
        let row_size = 2;
        let column_size = 2;
        let mat = Matrix :: new(vec![1.0,2.0,3.0,4.0],row_size,column_size).unwrap();
        let element = mat.get_element(1,2);
        assert_eq!(2.0, element);
        let element = mat.transpose().get_element(1,2);
        assert_eq!(3.0, element)
    }

    #[test]
    fn test_transpose() {
        let row_size = 2;
        let column_size = 2;
        let mat = Matrix :: new(vec![1.0,2.0,3.0,4.0],row_size,column_size).unwrap();
        let mat_t = mat.transpose().transpose();
        assert_eq!(mat_t,mat)
    }

    #[test]
    fn test_eigenvalues() {
        let mut mat = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3).unwrap();
        let w = svd(&mut mat);
        }

    #[test]
    fn test_singular_values() {
        let mut mat = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3).unwrap();
        let w = singular_values(&mut mat);
    }

    #[test]
    fn test_svd() {
        let mut mat = Matrix ::random(10,10);
        let w = singular_values(&mut mat);
    }

    #[test]
    fn test_tri() {
        let mut mat = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3).unwrap();
        let w =tril(&mut mat,0).ok();
    }

    #[test]
    fn test_add() {
        let mut mat = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3).unwrap();
        assert_eq!((&mat + &mat).ok(), matrix_map(&|&x| x + x, &mut mat).ok());
    }

    #[test]
    fn test_sub() {
        let mat = Matrix :: new(vec![3, 1, 1, 1, 3, 1, 1, 1, 3], 3, 3).unwrap();
        assert_eq!((&mat - &mat).ok(),Some(Matrix :: zeros(3,3)))
    }



    #[test]
    fn test_mul() {
        let mat = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3).unwrap();
        let ans = Matrix { elements: vec![9.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 9.0], row_size: 3, col_size: 3, transpose: false };
        assert_eq!((&mat * &mat).ok(), Some(ans))
    }


    #[test]
    fn test_lu_solve() {
        let mat = &mut Matrix :: random(10,10);
        let w = lufact(mat);
        let mut b =  Matrix :: random(10000,1);
        lusolve(w, &mut b);
    }

    #[test]
    fn test_dot(){
        let mut a = Matrix ::new(vec![1.0,2.0],2,1).unwrap();
        let mut b = Matrix ::new(vec![1.0,2.0],1,2).unwrap();
        let c = dot(&mut a,&mut b);
        // println!("{:?}", c.elements)
    }

    #[test]
    fn test_map(){
        let mut a : Matrix<f64>= Matrix ::random(10,10);
        let v = matrix_map(&|&x| x+x, &mut a);
        let e = matrix_map(&|&x| x*2.0, &mut a);
        assert_eq!(e.ok(),v.ok());
    }

    #[bench]
    fn bench_eig(ben : &mut Bencher){
        let i = 250;
        let mut mat = Matrix ::random(i,i);
        // let mut mat1= Matrix ::random(i,i);
        ben.iter( ||eigenvalues(&mut mat))
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

}

    // #[bench]
    // fn bench_lu_solve(ben : &mut Bencher){
    //     let mat = Matrix :: new(vec![-10.0,0.0,0.0,2.0],2,2);
    //     let mut k = mat.to_owned();
    //     let mut b =  Matrix :: new(vec![1.0,2.0],2,1);
    //     ben.iter( || lusolve(lufact(&mut k),&mut b))
    // }
