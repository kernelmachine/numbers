// rcond, rank,
//     det, invlndet,


use super::{Matrix,Norm, Condition};

use matrixerror::MatrixError;
use lapack::*;

pub fn norm(a : &Matrix<f64>, inorm : Norm) -> f64 {
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
// THRESHOLD RCOND TO DETERMINE INVERTIBILITY
pub fn rcond(lu : (&mut Matrix<f64>, Vec<i32>), inorm : Norm) -> (Result<f64, MatrixError>, Condition) {
    let nm = match inorm {
         Norm :: OneNorm => b'1',
         Norm :: InfinityNorm => b'I',
         Norm :: FrobeniusNorm => return (Err(MatrixError::LapackInputError),Condition::NA) ,
         Norm :: MaxAbsValue => return (Err(MatrixError::LapackInputError),Condition::NA)
    };
    let (a,ipiv) = lu;
    let n = a.col_size;
    let lda = a.row_size;
    let anorm = norm(&a, Norm::OneNorm);
    let mut rcond = 0.0;
    let mut work = vec![0.0; 4 * n];
    let mut iwork = vec![0;  n];
    let mut info = 0;
    dgecon(nm, n, &a.elements, lda, anorm, &mut rcond, &mut work, &mut iwork, &mut info);
    // BROKEN
    let cond = match rcond.gt(&5.0){
        false => Condition :: WellConditioned,
        true => {
            match rcond.le(&500000.0){
                true => Condition :: IllConditioned,
                false => Condition :: Singular,
            }
        }

    };


    match info {
        1 => (Err(MatrixError::LapackComputationError),cond),
        0 => (Ok(rcond.recip()), cond),
        -1 => (Err(MatrixError::LapackInputError),cond),
        _ => (Err(MatrixError::UnknownError),cond)
    }

}
