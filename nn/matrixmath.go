package nn

import (
	"errors"
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
	ErrNumColNotEqualRows   = errors.New("number of columns in first matrix and number of rows in second matrix are not equal, cannot multiply")
	ErrNotCorresponding     = errors.New("matrices are not the same sizes")
)

// Sigmoid function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Create takes number of rows and columns, returns zero valued matrix and an error
func Create(m, n int) (Mat, error) {
	if m == 0 || n == 0 {
		return nil, ErrZeroLengthColsOrRows
	}
	a := make(Mat, m)
	for i := range a {
		a[i] = make([]float64, n)
	}

	return a, nil
}

// CreateRandom takes number of rows and columns, returns random valued (uniform distribution) matrix and an error
func CreateRandom(m, n int, v float64) (Mat, error) {
	if m == 0 || n == 0 {
		return nil, ErrZeroLengthColsOrRows
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

	return a, nil
}

// Size takes matrix a, returns number of rows (m), columns (n) and error
func Size(a Mat) (int, int, error) {
	if a == nil {
		return 0, 0, ErrEmptyMatrix
	}

	return len(a), len(a[0]), nil
}

// Add adds two matrices, returns new matrix and error
func Add(a, b Mat) (Mat, error) {
	m, n, err := Size(a)
	if err != nil {
		return nil, err
	}
	m2, n2, err := Size(b)
	if err != nil {
		return nil, err
	}
	if m != m2 || n != n2 {
		return nil, ErrNotCorresponding
	}

	c, err := Create(m, n)
	if err != nil {
		return nil, err
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			c[i][j] = a[i][j] + b[i][j]
		}
	}
	return c, nil
}

// Subtract subtracts two matrices, returns new matrix and error
func Subtract(a, b Mat) (Mat, error) {
	m, n, err := Size(a)
	if err != nil {
		return nil, err
	}
	m2, n2, err := Size(b)
	if err != nil {
		return nil, err
	}
	if m != m2 || n != n2 {
		return nil, ErrNotCorresponding
	}

	c, err := Create(m, n)
	if err != nil {
		return nil, err
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			c[i][j] = a[i][j] - b[i][j]
		}
	}
	return c, nil
}

// Mul takes two matrices, a and b, returns product of their multiplication and an error
func Mul(a, b Mat) (Mat, error) {
	// Get sizes of matrix a and b
	n, m1, err := Size(a)
	if err != nil {
		return nil, err
	}
	m2, p, err := Size(b)
	if err != nil {
		return nil, err
	}
	// Number of columns in the first matrix must be equal number of rows in the second
	if m1 != m2 {
		return nil, ErrNumColNotEqualRows
	}

	c, err := Create(n, p)
	if err != nil {
		return nil, err
	}

	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			for r := 0; r < m1; r++ {
				c[i][j] += a[i][r] * b[r][j]
			}
		}
	}

	return c, nil
}

// ScalarMul takes matrix and a scalar float64, returns error
func ScalarMul(a Mat, c float64) error {
	m, n, err := Size(a)
	if err != nil {
		return err
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a[i][j] = c * a[i][j]
		}
	}
	return nil
}

// ScalarAdd takes matrix and a scalar float64, returns error
func ScalarAdd(a Mat, c float64) error {
	m, n, err := Size(a)
	if err != nil {
		return err
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a[i][j] += c
		}
	}
	return nil
}

// ApplyFunc applies function to a given matrix, returns error
func ApplyFunc(a Mat, f fn) error {
	m, n, err := Size(a)
	if err != nil {
		return err
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a[i][j] = f(a[i][j])
		}
	}
	return nil
}
