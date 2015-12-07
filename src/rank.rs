

use super::{Matrix,Norm, Condition, RectMat};

use matrixerror::MatrixError;
use lapack::*;
use factorizations::*;


/// Calculate the norm of a matrix (1-Norm, Infinity-Norm, Frobenius Norm, or Max Absolute Value)
///
/// # Arguments
///
/// * `Matrix` - Matrix of type f64
///
/// # Example
/// ```
///#[macro_use] extern crate numbers;
///use numbers::{Matrix, Norm};
/// pub fn main(){
/// let mut a = Matrix::new(vec![2.0,4.0,3.0,5.0], 2,2).ok().unwrap();
/// let n = numbers::rank::norm(&mut a, Norm::InfinityNorm);
/// assert!(8.0 - n < 1e-14)
///}
///```
pub fn norm<T:RectMat>(a : &T, inorm : Norm) -> f64 {
    let norm = match inorm {
         Norm :: OneNorm => b'1',
         Norm :: InfinityNorm => b'I',
         Norm :: FrobeniusNorm => b'F',
         Norm :: MaxAbsValue => b'M',
    };
    let m = a.col_size;
    let n = a.row_size;
    let lda = a.row_size;
    let mut work = vec![0.0; 4*n];
    dlange(norm, m, n, &a.elements,lda, &mut work)


}

/// Get the number of linearly independent rows or columns.
///
/// # Arguments
///
/// * `Matrix` - Matrix of type f64
///
/// # Example
/// ```
///#[macro_use] extern crate numbers;
///use numbers::Matrix;
/// pub fn main(){
/// let mut a = Matrix::new(vec![1.0,2.0,1.0,2.0], 2,2).ok().unwrap();
/// let n = numbers::rank::rank(&mut a).ok().unwrap();
/// assert_eq!(1, n)
///}
///```
pub fn rank<T: RectMat>(a : &mut T) -> Result<usize, MatrixError> {

    if let Ok(mut s) = singular_values(a) {
        let z = &mut s.elements;
        println!("{:?}",z);
        return Ok(z.iter().filter(|x| *x > &1e-4).collect::<Vec<&f64>>().len())
    }
    Ok(0 as usize)

}

/// Determine the condition of a matrix via the condition number.
///
/// # Arguments
///
/// * `Matrix` - Matrix of type f64
///
/// # Example
/// ```
///#[macro_use] extern crate numbers;
///use numbers::{Matrix, Norm, Condition};
/// pub fn main(){
/// let mut a = Matrix::new(vec![5.0,10.0,1.0,2.01], 2,2).ok().unwrap();
/// let c = numbers::rank::cond(&mut a, Norm :: InfinityNorm).ok().unwrap();
/// assert_eq!(c, Condition::IllConditioned)
///}
///```
pub fn cond<T: RectMat>(a : &mut T, inorm : Norm) -> Result<Condition, MatrixError>{
    let nm = match inorm {
         Norm :: OneNorm => b'1',
         Norm :: InfinityNorm => b'I',
         Norm :: FrobeniusNorm => return Err(MatrixError::LapackInputError),
         Norm :: MaxAbsValue => return Err(MatrixError::LapackInputError)
    };
    let cond = Condition :: NotAvailable;
    if let Ok((l, ipiv)) = lufact (a){
        let n = l.col_size;
        let lda = l.row_size;
        let anorm = norm(&l, Norm::OneNorm);
        let mut work = vec![0.0; 4 * n];
        let mut iwork = vec![0;  n];
        let mut rcond = 0.0;
        let mut info = 0;
        dgecon(nm, n, &l.elements, lda, anorm, &mut rcond, &mut work, &mut iwork, &mut info);
        let cond = match rcond.recip().gt(&1000.0){
            false => Condition :: WellConditioned,
            true => {
                match rcond.recip().le(&500000.0){
                    true => Condition :: IllConditioned,
                    false => Condition :: Singular,
                }
            }

        };

        match info {
            x if x > 0 => return Err(MatrixError::LapackComputationError),
            0 => return Ok(cond),
            x if x < 0 => return  Err(MatrixError::LapackInputError),
            _ => return Err(MatrixError::UnknownError),
        };

    }
    Ok(cond)
}
