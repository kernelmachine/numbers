use super::{Matrix,Triangular};
use blas::*;
use num::traits::Num;
use rand :: Rand;
use matrixerror::MatrixError;


pub fn dot (a : &mut Matrix<f64>, b : &mut Matrix<f64>) -> Result<Matrix<f64>, MatrixError>{
        let m = a.row_size;
        let n = b.col_size;

        if a.col_size != b.col_size {
            return Err(MatrixError::MismatchedDimensions)
        }
        let k = a.col_size;
        let mut c = vec![0.0; m*n];
        dgemm(b'N', b'N', m, n, k, 1.0, &a.elements, m, &b.elements,k, 0.0,&mut c, m);
        Ok(Matrix {
            elements : c,
            row_size : m,
            col_size : n,
            transpose : false,
        })


}

/// Maps a function closure onto a matrix.
    ///
    ///
    /// # Arguments
    ///
    /// * `func` - A function closure, e.g. &|&x| 2 * x
    /// * `a` - A matrix
    ///
    /// # Example
    /// ```
    /// let mut a : Matrix<f64>= Matrix :: random(10,10);
    /// let v = matrix_map(&|&x| x+x, &mut a);
    /// let e = matrix_map(&|&x| x*2.0, &mut a);
    /// assert_eq!(e.ok(),v.ok());
    /// ```
pub fn matrix_map <T: Num + Clone + Rand> (func : &Fn(&T) -> T, a : &mut Matrix<T>) -> Result<Matrix<T>, MatrixError>{
      Ok(Matrix {
           elements: a.elements.iter().map(func).collect(),
           row_size : a.row_size,
           col_size : a.col_size,
           transpose : false,
       })
}


pub fn triu<T : Num + Clone + Rand>(a: &mut Matrix<T>, k: usize ) -> Result<Matrix<T>, MatrixError>{
    let tri_mat = Matrix :: tri(a.row_size, a.col_size, k, Triangular::Upper);
    &tri_mat *&a
}

pub fn tril<T : Num + Clone + Rand>(a: &mut Matrix<T>, k: usize ) -> Result<Matrix<T>, MatrixError>{
    let tri_mat = Matrix :: tri(a.row_size, a.col_size, k, Triangular::Lower);
    &tri_mat *&a
}

pub fn inverse(a : &mut Matrix<f64> ) ->Result<Matrix<f64>,MatrixError> {
    unimplemented!() // via LU factorization

}

pub fn pseudoinverse(a : &mut Matrix<f64> ) ->Result<Matrix<f64>,MatrixError> {
    // take inverse of singualr vectors in SVD factorization
    unimplemented!()

}
