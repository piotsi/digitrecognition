package nn

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
)

// Mat represents matrix
type Mat [][]float64

type fn func(float64) float64

// These are errors
var (
	ErrEmptyMatrix          = errors.New("empty matrix")
	ErrZeroLengthColsOrRows = errors.New("zero length columns or row given")
	ErrNumColNotEqualRows   = errors.New("number of columns in first matrix and number of rows in second matrix is not equal, cannot multiply")
	ErrNotCorresponding     = errors.New("matrices are not the same sizes")
)

// Sigmoid function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// SigmoidPrime function sig*(1-sig)
func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

// Create takes number of rows and columns, returns zero valued matrix and an error
func Create(m, n int) Mat {
	if m == 0 || n == 0 {
		log.Panic(ErrZeroLengthColsOrRows)
	}
	a := make(Mat, m)
	for i := range a {
		a[i] = make([]float64, n)
	}

	return a
}

// CreateRandom takes number of rows and columns, returns random valued (uniform distribution) matrix and an error
func CreateRandom(m, n int, v float64) Mat {
	if m == 0 || n == 0 {
		log.Panic(ErrZeroLengthColsOrRows)
	}
	a := make(Mat, m)

	min := -1 / math.Sqrt(v)
	max := 1 / math.Sqrt(v)

	for i := range a {
		a[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			a[i][j] = min + rand.Float64()*(max-min)
		}
	}

	return a
}

// Size takes matrix a, returns number of rows (m), columns (n) and error
func Size(a Mat) (int, int) {
	if a == nil {
		log.Panic(ErrEmptyMatrix)
	}

	return len(a), len(a[0])
}

// Add adds two matrices, returns new matrix and error
func Add(a, b Mat) Mat {
	m, n := Size(a)
	m2, n2 := Size(b)
	if m != m2 || n != n2 {
		fmt.Println(m, m2, n, n2)
		log.Panic(ErrNotCorresponding)
	}

	c := Create(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			c[i][j] = a[i][j] + b[i][j]
		}
	}
	return c
}

// Subtract subtracts two matrices, returns new matrix and error
func Subtract(a, b Mat) Mat {
	m, n := Size(a)
	m2, n2 := Size(b)

	if m != m2 || n != n2 {
		log.Panic(ErrNotCorresponding)
	}

	c := Create(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			c[i][j] = a[i][j] - b[i][j]
		}
	}
	return c
}

// Mul takes two matrices, a and b, returns product of their multiplication and an error
func Mul(a, b Mat) Mat {
	// Get sizes of matrix a and b
	n, m1 := Size(a)
	m2, p := Size(b)

	// Number of columns in the first matrix must be equal number of rows in the second
	if n != m2 || m1 != p {
		log.Panic(ErrNumColNotEqualRows)
	}

	c := Create(n, p)
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			c[i][j] += a[i][j] * b[i][j]
		}
	}

	return c
}

// Dot takes two matrices, a and b, returns their dot product of their multiplication and an error
func Dot(a, b Mat) Mat {
	// Get sizes of matrix a and b
	n, m1 := Size(a)
	m2, p := Size(b)

	// Number of columns in the first matrix must be equal number of rows in the second
	if m1 != m2 {
		log.Panic(ErrNumColNotEqualRows)
	}

	c := Create(n, p)

	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			for r := 0; r < m1; r++ {
				c[i][j] += a[i][r] * b[r][j]
			}
		}
	}

	return c
}

// ScalarMul takes matrix and a scalar float64, returns error
func ScalarMul(a Mat, c float64) Mat {
	m, n := Size(a)
	b := Create(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			b[i][j] = c * a[i][j]
		}
	}
	return b
}

// ScalarAdd takes matrix and a scalar float64, returns error
func ScalarAdd(a Mat, c float64) {
	m, n := Size(a)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a[i][j] += c
		}
	}
}

// ApplyFunc applies function to a given matrix, returns new Mat and error
func ApplyFunc(a Mat, f fn) Mat {
	m, n := Size(a)
	b := Create(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			b[i][j] = f(a[i][j])
		}
	}
	return b
}

// Transpose transposes matrix a, returns new matrix b and error
func Transpose(a Mat) Mat {
	m, n := Size(a)
	b := Create(n, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			b[j][i] = a[i][j]
		}
	}
	return b
}

// SliceToMat takes 1D slice, returns 2D Mat
func SliceToMat(a []float64) Mat {
	l := len(a)
	b := Create(l, 1)
	for i := 0; i < l; i++ {
		b[i][0] = a[i]
	}
	return b
}
