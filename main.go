package main

import (
	nn "digitrecognition/nn"
	"fmt"
	"math/rand"
	"time"
	// draw "github.com/piotsik/draw"
)

func main() {
	// nn.PNGtoCSV()
	// records := make(chan []string)
	// go nn.OpenMNIST(records, 10, "train")
	// nn.DrawDigit(records)
	// draw.Run()
	rand.Seed(time.Now().UnixNano())
	start := time.Now()

	a, _ := nn.CreateRandom(2, 1, 1)
	b, _ := nn.CreateRandom(2, 1, 1)
	fmt.Println("a: ", a)
	fmt.Println("b: ", b)
	_ = nn.ApplyFunc(b, nn.Sigmoid)
	fmt.Println("b: ", b)
	_ = nn.ScalarMul(b, 0.1)
	fmt.Println("b: ", b)
	_ = nn.ScalarAdd(b, 1.0)
	fmt.Println("b: ", b)

	fmt.Printf("Time taken: %s", time.Since(start))

	// time.Sleep(time.Millisecond)
}
