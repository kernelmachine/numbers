use super::{Matrix, Eig, Triangular, Trans, RectMat};
use lapack::*;
use matrixerror::MatrixError;



/// Get the eigenvalues of a general matrix.
///
/// # Arguments
///
/// * `Matrix` - Matrix of type f64
///
/// # Example
/// ```
///#[macro_use] extern crate numbers;
///use numbers::{Matrix, Eig, Triangular};
/// pub fn main(){
/// let mut a = Matrix::new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3,3).ok().unwrap();
/// let eigs = numbers::eigenvalues::eigenvalues(&mut a, Eig::Eigenvalues, Triangular::Upper).ok().unwrap();
/// let ans = Matrix::new(vec![2.0, 2.0, 5.0], 3,1).ok().unwrap();
/// matrix_equal!(eigs,ans)
///}
///```
pub fn eigenvalues <T: RectMat> (a : &mut T, eorv : Eig, tri : Triangular) -> Result<T,MatrixError>{
    let n = a.col_size;
    let mut w = vec![0.0; n];
    let mut work = vec![0.0; 4 * n];
    let lwork = 4 * n as isize;
    let mut info = 0;
    let e = match eorv {
        Eig::Eigenvalues => b'V',
        Eig::EigenvaluesAndEigenvectors => b'E'
    };

    let t = match tri {
        Triangular::Upper => b'U',
        Triangular::Lower => b'L'
    };

    dsyev(e, t, n, &mut a.elements, n, &mut w, &mut work, lwork, &mut info);

    match info {
        x if x > 0 => Err(MatrixError::LapackComputationError),
        0 => Ok (Matrix {
            elements : w.to_owned(),
            row_size : w.len(),
            col_size : 1,
            transpose : Trans :: Regular,
        }),
        x if x < 0 => Err(MatrixError::LapackInputError),
        _ => Err(MatrixError::UnknownError)
    }

}
