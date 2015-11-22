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

/// Generic Matrix object.
#[derive(Debug, Clone)]
pub struct Matrix <T : Num + Clone + Rand> {
    pub elements : Vec<T>,
    pub row_size : usize,
    pub col_size : usize,
    pub transpose :  bool,
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
    NA
}

/// Type declaration of Singular Value Decomposition Output
pub type SVD = (Matrix<f64>, Matrix<f64>, Matrix<f64>);

impl<'a, 'b, T : Num + Clone + Rand> Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Result<Matrix<T>, MatrixError>;

    fn add(self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.col_size != other.col_size{
             return Err(MatrixError::MismatchedDimensions)
         }


            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x+y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : false})

    }
}

impl<'a, 'b, T : Num + Clone + Rand> Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Result<Matrix<T>, MatrixError>;

    fn sub(self, other: &'b Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.col_size != other.col_size{
             return Err(MatrixError::MismatchedDimensions)
         }
        Ok(Matrix {
        elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x-y).collect(),
        row_size : self.row_size,
        col_size : other.col_size,
        transpose : false})
    }

}

impl<'a, 'b, T : Num + Clone + Rand> Div<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Result<Matrix<T>, MatrixError>;

    fn div(self, other: &'b Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.col_size != other.col_size{
             return Err(MatrixError::MismatchedDimensions)
         }
            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x/y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : false}
        )
        }
    }




impl<'a, 'b, T : Num + Clone + Rand> Mul <&'b Matrix<T>> for &'a Matrix<T> {
    type Output =Result<Matrix<T>, MatrixError>;

    fn mul(self, other: &'b Matrix<T>) ->Result<Matrix<T>, MatrixError>{


        if self.col_size != other.col_size{
             return Err(MatrixError::MismatchedDimensions)
         }

            Ok(Matrix {
            elements : self.elements.clone().zip_with(other.elements.clone(), |x,y| x*y).collect(),
            row_size : self.row_size,
            col_size : other.col_size,
            transpose : false})

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

    /// Create a new matrix
    pub fn new(e : Vec<T>, r_size : usize, c_size : usize) -> Result<Matrix<T>, MatrixError>{
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

    /// Create a matrix of zeros
    pub fn zeros (r_size : usize, c_size : usize) -> Matrix<T>{
        Matrix {
            elements : vec![Zero::zero();r_size*c_size],
            row_size : r_size,
            col_size : c_size,
            transpose : false,
        }
    }

    // Create a matrix of random elements.
    pub fn random(r_size : usize, c_size : usize) -> Matrix<T>{
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

    /// Create a square diagonal matrix from a vector of elements.
    pub fn diag_mat (a : Vec<T>) -> Matrix<T> {
        let mut mat = Matrix :: zeros(a.len(),a.len());
        for i in 1..a.len()+1{
            let e = &a[i-1];
            mat.replace(i, i, e.to_owned());
        }
        mat
    }

    /// Create an square identity matrix of specified row size.
    pub fn identity(row_size : usize) -> Matrix<T> {
          Matrix :: diag_mat(vec![One::one(); row_size])
    }

    /// Replace a value in the matrix at specified row and column indices.
    pub fn replace(&mut self,row: usize, col:usize, value : T) -> () {
        let ind = self.get_ind(row,col);
        self.elements[ind] = value;
    }

    /// Map an index in a matrix to index in its corresponding 1-d vector.
    pub fn get_ind(&self, row :usize, col : usize) -> usize{
        if self.transpose == true{
            return (row-1)+ (col-1)*self.row_size
        }
        (col-1) +(row-1)*self.col_size
    }

    /// Get an element from the matrix.
    pub fn get_element(&self, row : usize, col : usize) -> T{
        let elem = &self.elements[self.get_ind(row,col)];
        elem.to_owned()
    }


    /// Transpose the matrix. We're not actually changing anything in memory;
    // we just flag the matrix as transpose to change pointer reference to elements.
    pub fn transpose(&self) -> Matrix<T> {
        Matrix {
            elements : self.elements.clone(),
            row_size : self.col_size,
            col_size : self.row_size,
            transpose : match self.transpose { true => false, false => true}
        }
    }

    /// Get the diagonal of a matrix.
    pub fn diagonal (&self) -> Vec<T>{
        let mut diag : Vec<T> = Vec :: new();
        for elem in 1..min(self.row_size,self.col_size){
            diag.push(self.get_element(elem,elem));
        }
        diag
    }

    /// Get a triangular matrix with specified dimensions above or below specified diagonal.
    pub fn tri (row_size:usize, col_size : usize, k : usize, upper_or_lower : Triangular) -> Matrix<T>{
        let mut mat : Matrix<T> = Matrix :: zeros(row_size, col_size);
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



    /// Get a submatrix from a matrix.
    pub fn submatrix(&self, start : (usize,usize), dim : (usize,usize)) -> Matrix<T> {
    unimplemented!();
    }



}
