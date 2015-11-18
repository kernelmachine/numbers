use std::error;
use std::fmt;

#[derive(Debug)]
pub enum MatrixError{
    MismatchedDimensions,
    MalformedMatrix,
    ZeroDeterminant,
}

impl fmt::Display for MatrixError{

    fn fmt(&self, f : &mut fmt::Formatter ) -> fmt::Result{

        match *self{
            MatrixError::MismatchedDimensions=> write!(f, "Operation cannot be performed. Mismatched dimensions."),
            MatrixError::MalformedMatrix => write!(f, "Matrix is malformed."),
            MatrixError::ZeroDeterminant => write!(f, "Operation cannot be performed. Matrix has zero determinant.")
        }
    }
}

impl error::Error for MatrixError{

    fn  description(&self) -> &str {
        match *self{

            MatrixError::MismatchedDimensions => "Operation cannot be performed. Mismatched dimensions.",
            MatrixError::MalformedMatrix => "Matrix is malformed.",
            MatrixError::ZeroDeterminant => "Operation cannot be performed. Matrix has zero determinant."
        }

    }

    fn cause(&self) -> Option<&error::Error> {
      match *self {
          MatrixError::MismatchedDimensions => None,
          MatrixError::MalformedMatrix => None,
          MatrixError::ZeroDeterminant => None,

      }
  }


}
