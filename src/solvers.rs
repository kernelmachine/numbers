use super::{Matrix, Trans, RectMat};
use lapack::*;
use factorizations::*;
use matrixerror::MatrixError;


/// Solve Ax = b via LU Factorization.
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
/// let mut a = Matrix::new(vec![5.0, -1.0, 2.0, 10.0, 3.0, 7.0, 15.0, 17.0,19.0], 3,3).ok().unwrap();
/// let mut b = Matrix::new(vec![4.0,19.0,-6.0],3,1).ok().unwrap();
/// let mut x = numbers::solvers::lusolve(&mut a,&mut b).ok().unwrap();
/// let ans = numbers::operations::dot(&mut a, &mut x).ok().unwrap();
/// matrix_equal!(ans,b)
///}
///```
pub fn lusolve<T: RectMat>(a : &mut T, b : &mut T) ->  Result<T,MatrixError>{
    let (a,ipiv) = try!(lufact(a));
    let lda = a.row_size;
    let n = a.col_size;
    let ldb = b.row_size;
    let nrhs = b.col_size;
    let mut info = 0;

    dgetrs(b'N', n, nrhs, &a.elements, lda, &ipiv, &mut b.elements, ldb , &mut info);

    match info {
        x if x > 0 => Err(MatrixError::LapackComputationError),
        0 => Ok(Matrix {
            elements : b.elements.to_owned(),
            row_size : ldb,
            col_size : nrhs,
            transpose : Trans :: Regular,
        }),
        x if x < 0 => Err(MatrixError::LapackInputError),
        _ => Err(MatrixError::UnknownError)
    }


}
