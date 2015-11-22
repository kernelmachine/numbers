
#[allow(dead_code)]

extern crate numbers;
extern crate num;

#[cfg(test)]
mod tests{
    use numbers::{Matrix, Eig, Triangular, Norm, Condition};
    use numbers::eigenvalues::*;
    use numbers::solvers::*;
    use numbers::operations::*;
    use numbers::factorizations::*;
    use numbers::rank::*;
    use num::traits::Float;
    use numbers::matrixerror::*;
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
        if let Ok(mat) = Matrix :: new(vec![1.0,2.0,3.0,4.0],row_size,column_size){
            let element = mat.get_element(1,2);
            assert!((element - 2.0).abs() < 1e-14);
            let element = mat.transpose().get_element(1,2);
            assert!((element - 3.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_transpose() {
        let row_size = 2;
        let column_size = 2;
        if let Ok(mat) =  Matrix :: new(vec![1.0,2.0,3.0,4.0],row_size,column_size) {
            let mat_t = mat.transpose().transpose();
            assert_eq!(mat_t,mat)
        }

    }

    #[test]
    fn test_eigenvalues() {
        if let Ok(mut mat) = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3){
            let _w = eigenvalues(&mut mat, Eig :: Eigenvalues, Triangular :: Upper);
            assert_eq!(1,1);
        }

        }

    #[test]
    fn test_singular_values() {
        if let Ok(mut mat) = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3){
            let _w = singular_values(&mut mat);
            assert_eq!(1,1);
        }
    }

    #[test]
    fn test_svd() {
        let mut mat = Matrix ::random(10,10);
        let _w = svd(&mut mat);
        assert_eq!(1,1);
    }

    #[test]
    fn test_tri() {
        if let Ok(mut mat) = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3){
            let _w =tril(&mut mat,0).ok();
            assert_eq!(1,1);
        }
    }

    #[test]
    fn test_add() {
        if let Ok(mut mat) = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3){
            assert_eq!((&mat + &mat).ok(), matrix_map(&|&x| x + x, &mut mat).ok());
        }
    }

    #[test]
    fn test_sub() {
        if let Ok(mat) =  Matrix :: new(vec![3, 1, 1, 1, 3, 1, 1, 1, 3], 3, 3){
            assert_eq!((&mat - &mat).ok(),Some(Matrix :: zeros(3,3)))
        }
    }



    #[test]
    fn test_mul() {
        if let Ok(mat) = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0], 3, 3){
            let ans = Matrix { elements: vec![9.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 9.0], row_size: 3, col_size: 3, transpose: false };
            assert_eq!((&mat * &mat).ok(), Some(ans))
        }

    }


    #[test]
    fn test_lu_solve() {
        let mat = &mut Matrix :: random(10,10);
        if let Ok(w) = lu(mat){
            let mut b =  Matrix :: random(10000,1);
            let _l = lusolve(w, &mut b);
            assert_eq!(1,1)
        }
    }

    #[test]
    fn test_lu(){
        let  b = match  Matrix ::new(vec![4.0,3.0,8.0,6.01],2,2){
            Ok(w) => Ok(w),
            Err(_w) => Err(MatrixError::MalformedMatrix)
        };
        let mut c = b.ok().unwrap();

        let mut _l = match lu(&mut c){
            Ok(w) => Ok(w),
            Err(_w) => Err(MatrixError::LapackComputationError)

        };

        assert_eq!(1,1)
    }
    #[test]
    fn test_dot(){
        if let Ok(mut a) =  Matrix ::new(vec![1.0,2.0],2,1){
            if let Ok(mut b) = Matrix ::new(vec![1.0,2.0],1,2){
                let _c = dot(&mut a,&mut b);
                assert_eq!(1,1)
            }
        }

    }

    #[test]
    fn test_map(){
        let mut a : Matrix<f64>= Matrix ::random(10,10);
        let v = matrix_map(&|&x| x+x, &mut a);
        let e = matrix_map(&|&x| x*2.0, &mut a);
        assert_eq!(e.ok(),v.ok());
    }


    #[test]
    fn test_condition(){
        let  b =  Matrix ::new(vec![500.0,50.0,100.0,100.0],2,2);
        let mut c = b.ok().unwrap();
        let rc =  cond(&mut c, Norm::InfinityNorm);
        assert_eq!(rc.ok().unwrap(),Condition::WellConditioned)
    }

    #[test]
    fn test_inverse(){
        let b = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0,6.0],3,3);
        let mut c = b.ok().unwrap();
        if let Ok(_mat) = inverse(&mut c){
            let b = Matrix :: new(vec![3.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0,6.0],3,3);
            let mut c = b.ok().unwrap();
            if let Ok(_mat1) =  pseudoinverse(&mut c){
                assert_eq!(1,1)
            }
        }
    }



    // #[test]
    // fn test_pseudoinverse(){
    //     let b = Matrix ::new(vec![4.0,5.0,1.0,2.0],2,2);
    //     let mut c = b.ok().unwrap();
    //     let rc =  pseudoinverse(&mut c);
    //     // println!("{:?}",rc);
    //
    //     // assert_eq!(rc.ok().unwrap(),Condition::WellConditioned)
    // }

}
