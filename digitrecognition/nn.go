package main // Change to digitrecognition when finished

import (
	"fmt"
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

// InitNN initializes neural network
func InitNN() int {
	rand.Seed(time.Now().UnixNano()) // Initialize global Source of pseud-random values
	var net Data

	// First layer must be same size as input size!
	// Perceptron model
	net.createModel(5, 5, 10)

	// --------------------------------
	inpData := []float64{1, 1, 1, 1, 1}
	var inp = mat.NewDense(len(inpData), 1, inpData)

	prediction := net.predict(inp)

	fa := mat.Formatted(prediction, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("mat % .2f\n\n", fa)
	return 0 // change to return recognized digit
}

// ----------------------------------------------------------------
// Create model data: layers, weights, biases
func (net *Data) createModel(ls ...int32) {
	// Define layer sizes (neurons in each layer)
	net.layerSizes = ls

	// Define weight matrix sizes (m by n synapses between each layer)
	for i := 1; i < len(net.layerSizes); i++ {
		net.weightShapes = append(net.weightShapes, []int32{net.layerSizes[i], net.layerSizes[i-1]})
	}

	// Fill each weight matrix with normally distributed value
	net.weights = make([]*mat.Dense, len(net.weightShapes))
	for i, ws := range net.weightShapes {
		data := make([]float64, ws[0]*ws[1])
		for j := range data {
			data[j] = rand.NormFloat64() / math.Pow(float64(ws[1]), 0.5)
		}
		net.weights[i] = mat.NewDense(int(ws[0]), int(ws[1]), data)
	}

	// Create biases matrices filled with zeros
	net.biases = make([]*mat.Dense, len(net.weightShapes))
	for i, ls := range net.layerSizes {
		if i > 0 {
			zeros := make([]float64, ls)
			net.biases[i-1] = mat.NewDense(int(ls), 1, zeros)
		}
	}
}

// Predict?
func (net *Data) predict(input mat.Matrix) mat.Matrix {
	for i := range net.weights {
		dot := matmul(net.weights[i], input)
		sum := add(dot, net.biases[i])
		input = f(activation, sum)
	}
	return input
}

// ----------------------------------------------------------------
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
	c := mat.NewDense(m, n, nil) // The result matrix has to have the m rows of the first matrix and n columns of the second
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

// Remove this after production of nn.go to run *.go
// Seperate training and predicting
func main() {
	NN()
}
