package digitrecognition

import (
	"math"
	"time"

	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Data holds information about the model
type Data struct {
	layerSizes   []int32
	weightShapes [][]int32
	weights      []*mat.Dense
	biases       []*mat.Dense
}

// Remove this after production of nn.go to run *.go
func main() {
	NN()
}

// NN is a neural network
func NN() {
	rand.Seed(time.Now().UnixNano()) // Initialize global Source of pseud-random values
	var d Data
	d.gcreateModel(784, 5, 10)

	// adata := []float64{1, 2, 3, 4}
	// bdata := []float64{1, 2, 3, 4}
	// a := mat.NewDense(2, 2, adata)
	// b := mat.NewDense(2, 2, bdata)
	// c :=
	// fmt.Printf("% v", matmul())

}

func (d *Data) createModel(ls ...int32) {
	// Define layer sizes (neurons in each layer)
	d.layerSizes = ls

	// Define weight matrix sizes (m by n synapses between each layer)
	for i := 1; i < len(d.layerSizes); i++ {
		d.weightShapes = append(d.weightShapes, []int32{d.layerSizes[i], d.layerSizes[i-1]})
	}

	// Fill each weight matrix with normally distributed value
	d.weights = make([]*mat.Dense, len(d.weightShapes))
	for i, ws := range d.weightShapes {
		data := make([]float64, ws[0]*ws[1])
		for j := range data {
			data[j] = rand.NormFloat64()
		}
		d.weights[i] = mat.NewDense(int(ws[0]), int(ws[1]), data)
	}

	// Create biases matrices filled with zeros
	d.biases = make([]*mat.Dense, len(d.weightShapes))
	for i, ls := range d.layerSizes {
		if i > 0 {
			zeros := make([]float64, ls)
			d.biases[i-1] = mat.NewDense(int(ls), 1, zeros)
		}
	}
}

// def (d *Data) predict() {
// 	return
// }

// Add two matrices
func add(a, b mat.Matrix) mat.Matrix {
	m, n := a.Dims()             // Get dimensions of first matrix
	c := mat.NewDense(m, n, nil) // Create empty matrix receiver with m by n size
	c.Add(a, b)                  // Add a and b matrices and place result into receiver
	return c
}

// Multiply two matrices
func matmul(a, b mat.Matrix) mat.Matrix {
	m, _ := a.Dims()             // Get dimensions of first matrix
	_, n := b.Dims()             // Get dimensions of second matrix
	c := mat.NewDense(m, n, nil) // The result matrix has to have the m columns of the first matrix and n rows of the second
	c.Product(a, b)              // Multiply two matrices and place result into receiver
	return c
}

// Apply function to a matrix
func f(fn func(m, n int, x float64) float64, a mat.Matrix) mat.Matrix {
	m, n := a.Dims()             // Get dimensions of first matrix
	c := mat.NewDense(m, n, nil) // Create empty matrix receiver with m by n size
	c.Apply(fn, a)               // Apply function on matrix and place result into receiver
	return c
}

// Gonums' Apply function needs m and n
func activation(m, n int, x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x)) // Sigmoid function
}
