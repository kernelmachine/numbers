//! Numbers provides safe, modular, and fast high-dimensional matrix computations.

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_must_use)]
#![feature(custom_derive)]
#![feature(test)]
#![feature(plugin)]
#![plugin(clippy)]
#![allow(match_bool)]

extern crate blas;
extern crate lapack;
extern crate rand;
extern crate zipWith;
extern crate num;
extern crate test;

pub mod matrixerror;
pub mod operations;
pub mod eigenvalues;
pub mod solvers;
pub mod factorizations;
pub mod rank;

use matrixerror::MatrixError;
use rand::{thread_rng, Rng, Rand};
use std::cmp::*;
use std::ops::{Add, Sub, Mul, Div};
use num::traits::{Num, Zero, One};
use zipWith::IntoZipWith;


enum MatrixTypes<T, U, R> where
    T: RectMat,
    U: SqMat,
    R: NonSingularMat
{
    Rectangle(T),
    Square(U),
    NonSingularMat(R)
}

trait RectMat : Num + Rand + Clone{
    fn new(e : Vec<f64>, r_size : usize, c_size : usize) -> Result<   Self, MatrixError>;
    fn random(r_size : usize, c_size: usize) -> Self ;
    fn replace(&mut self,row: usize, col:usize, value :f64) -> ();
    fn zeros (r_size : usize, c_size : usize) -> Self;
    fn diag_mat (a : Vec<f64>) -> Self;
    fn get_element(&self, row : usize, col : usize) -> f64;
    fn get_ind(&self, row :usize, col : usize) -> usize;
    fn transpose(&self) -> Self;
    fn diagonal (&self) -> Vec<f64>;
    fn tri (row_size:usize, col_size : usize, k : usize, upper_or_lower : Triangular) -> Result<Self,MatrixError>;
}

trait SqMat: RectMat{
    fn new(e : Vec<f64>, r_size : usize, c_size : usize) -> Result<   Self, MatrixError>;
    fn diag_mat (a : Vec<f64>) -> Self;
    fn identity(row_size : usize) -> Self;
    fn pseudoinverse(a: &mut Self) ->Result<Self,MatrixError>;
}

trait NonSingularMat : SqMat{
    fn inverse(a : &mut Self ) ->Result<Self,MatrixError>;
}


/// Generic Matrix object.
#[derive(Debug, Clone, Rand, Num)]
pub struct Matrix {
    pub elements : Vec<f64>,
    pub row_size : usize,
    pub col_size : usize,
    pub transpose : Trans,
}



impl RectMat for Matrix{

    /// Create rectangular matrix
    fn new(e : Vec<f64>, r_size : usize, c_size : usize) -> Result<Matrix, MatrixError>{
        if r_size * c_size != e.len(){
            return Err(MatrixError :: MalformedMatrix)
        }
        Ok(Matrix {
            elements : e,
            row_size : r_size,
            col_size : c_size,
            transpose : Trans :: Regular,
        })
    }

    fn random(r_size : usize, c_size : usize) -> Matrix{
        let e = rand::thread_rng()
        .gen_iter::<f64>()
        .take(r_size*c_size)
        .collect::<Vec<f64>>();

        Matrix {
            elements : e,
            row_size : r_size,
            col_size : c_size,
            transpose : Trans :: Regular,
        }
    }

    fn replace(&mut self,row: usize, col:usize, value : f64) -> () {
        let ind = self.get_ind(row,col);
        self.elements[ind] = value;
    }

    /// Create matrix with all zeros.
    fn zeros (r_size : usize, c_size : usize) -> Matrix{
        Matrix {
            elements : vec![Zero::zero();r_size*c_size],
            row_size : r_size,
            col_size : c_size,
            transpose : Trans :: Regular,
        }
    }

    /// Map an index in a matrix to index in its corresponding 1-d vector.
    fn get_ind(&self, row :usize, col : usize) -> usize{
        if self.transpose == Trans :: Transpose{
            return (row-1)+ (col-1)*self.row_size
        }
        (col-1) +(row-1)*self.col_size
    }

    /// Get an element from the matrix.
    fn get_element(&self, row : usize, col : usize) -> f64{
        let elem = &self.elements[self.get_ind(row,col)];
        elem.to_owned()
    }
    /// Transpose the matrix.
    /// We're not actually changing anything in memory; we just flag the matrix as transpose to change pointer reference to elements.
    fn transpose(&self) -> Matrix {
        Matrix {
            elements : self.elements.clone(),
            row_size : self.col_size,
            col_size : self.row_size,
            transpose : match self.transpose { Trans :: Transpose => Trans :: Regular, Trans :: Regular => Trans :: Transpose}
        }
    }

    /// Get the diagonal of a matrix.
    fn diagonal (&self) -> Vec<f64>{
        let mut diag : Vec<f64> = Vec :: new();
        for elem in 1..min(self.row_size,self.col_size){
            diag.push(self.get_element(elem,elem));
        }
        diag
    }

    /// Get a triangular matrix with specified dimensions above or below specified diagonal.
    fn tri (row_size:usize, col_size : usize, k : usize, upper_or_lower : Triangular) -> Matrix{
        let mut mat : Matrix = Matrix :: zeros(row_size, col_size);
        match k {
        x if x > row_size => Err(MatrixError :: IndexError),
        x if x > col_size => Err(MatrixError :: IndexError),
        }
        for i in 1..row_size+1{
            for j in 1..col_size+1{
                match upper_or_lower{

                    Triangular::Upper =>{
                        if i <= j + k{
                            mat.replace(i,j,One::one());
                        }
                    }
                    Triangular::Lower => {
                        if i >= j + k{
                            mat.replace(i,j,One::one());
                        }
                    }

                }
            }
        }
        mat
    }

}

impl SqMat for Matrix {

        /// Create square matrix
        fn new(e : Vec<f64>, rc_size : usize) -> Result<Matrix, MatrixError>{
            if rc_size * rc_size != e.len(){
                return Err(MatrixError :: MalformedMatrix)
            }
            Ok(Matrix {
                elements : e,
                row_size : rc_size,
                col_size : rc_size,
                transpose : Trans :: Regular,
            })
        }

        /// Create a diagonal matrix of specified row size.

        fn diag_mat (a : Vec<f64>) -> Matrix {
            let mut mat = Matrix :: zeros(a.len(),a.len());
            for i in 1..a.len()+1{
                let e = &a[i-1];
                mat.replace(i, i, e.to_owned());
            }
            mat
        }

        /// Create an square identity matrix of specified row size.
        fn identity(row_size : usize) -> Matrix {
              Matrix :: diag_mat(vec![One::one(); row_size])
        }

        fn pseudoinverse(a: &mut Matrix) ->Result<Matrix,MatrixError> {
            let (mut u,mut e, mut vt) = try!(factorizations::svd(a));
            let inv_e = try!(a.matrix_map(&|x : &f64| if x > &0.0 { x.recip()} else { 0.0 },&mut e));
            let mut d = try!(operations::dot(&mut u, &mut inv_e.transpose()));
            operations::dot (&mut d, &mut vt)
        }

}

impl NonSingularMat for Matrix {

    fn inverse(a : &mut Matrix ) ->Result<MatrixError,MatrixError> {

            let (l, ipiv) = factorizations::lufact (a);
            let n = l.col_size;
            let lda = l.row_size;
            let mut work = vec![0.0; 4 * n];
            let iwork = n as isize * 4;
            let mut info = 0;
            lapack::dgetri(n, &mut l.elements, lda, &ipiv,&mut work, iwork, &mut info);
            match info {
                x if x > 0 => return Err(MatrixError::LapackComputationError),
                0 => { let m = Ok(l.to_owned()); return m}
                x if x < 0 => return  Err(MatrixError::LapackInputError),
                _ => return Err(MatrixError::UnknownError),
            };

        Err(MatrixError::SingularMatrix)
    }

}

/// Triangular represents two variants of triangular matrices : Upper and Lower. Use this enum
/// for functions that require specification of a triangular matrix output.
#[derive(Debug, Clone)]
pub enum Triangular{
    Upper,
    Lower
}

/// Norm provides four variants of matrix norms.
#[derive(Debug, Clone)]
pub enum Norm{
    OneNorm,
    InfinityNorm,
    FrobeniusNorm,
    MaxAbsValue
}
/// For relevant functions, Eig represents the option to output either just eigenvalues or
/// both eigenvalues and eigenvectors.
#[derive(Debug, Clone)]
pub enum Eig {
    Eigenvalues,
    EigenvaluesAndEigenvectors,
}

/// Condition provides four matrix conditions to determine the feasibility of inversion
#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    WellConditioned,
    IllConditioned,
    Singular,
    NotAvailable
}

/// Trans enumerates transpose/non-transpose matrices
#[derive(Debug, Clone , PartialEq)]
pub enum Trans {
    Transpose,
    Regular
}

/// Type declaration of Singular Value Decomposition Output
pub type SVD = (Matrix, Matrix, Matrix);



impl<'a, 'b> Add<&'b Matrix> for &'a Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn add(self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.col_size != other.col_size{
             return Err(MatrixError::MismatchedDimensions)
         }


            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x+y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : Trans :: Regular})

    }
}

impl<'a, 'b> Sub<&'b Matrix> for &'a Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn sub(self, other: &'b Matrix) -> Result<Matrix, MatrixError> {
        if self.col_size != other.col_size{
             return Err(MatrixError::MismatchedDimensions)
         }
        Ok(Matrix {
        elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x-y).collect(),
        row_size : self.row_size,
        col_size : other.col_size,
        transpose : Trans :: Regular})
    }

}

impl<'a, 'b> Div<&'b Matrix> for &'a Matrix {
    type Output = Result<Matrix, MatrixError>;

    fn div(self, other: &'b Matrix) -> Result<Matrix, MatrixError> {
        if self.col_size != other.col_size{
             return Err(MatrixError::MismatchedDimensions)
         }
            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x/y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : Trans :: Regular}
        )
        }
    }




impl<'a, 'b> Mul <&'b Matrix> for &'a Matrix {
    type Output =Result<Matrix, MatrixError>;

    fn mul(self, other: &'b Matrix) ->Result<Matrix, MatrixError>{


        if self.col_size != other.col_size{
             return Err(MatrixError::MismatchedDimensions)
         }

            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x*y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : Trans :: Regular})

        }
    }


impl PartialEq for Matrix{
    fn eq(&self, other: &Matrix) -> bool {
         (self.elements == other.elements)
       & (self.row_size == other.row_size)
       & (self.col_size == other.col_size)
       & (self.transpose  == other.transpose)
   }
}

#[macro_export]
macro_rules! matrix_equal {
    ( $x:expr , $y : expr) => (
        {
            assert!($x.row_size == $y.row_size);
            assert!($x.col_size == $y.col_size);
            assert!($x.transpose  == $y.transpose);
            if let Ok(s) = &$x - &$y{
                for elem in s.elements{
                    assert!(elem.abs() < 1e-6)
                }
            }
        }

    )
    }
