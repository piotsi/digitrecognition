package main

import (
	nn "digitrecognition/nn"
	"flag"
	"fmt"
	"math/rand"
	"time"
	// draw "github.com/piotsik/draw"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	start := time.Now()

	net := nn.Construct(784, 200, 10, 0.1)

	mode := flag.String("m", "", "a string")
	flag.Parse()

	switch *mode {
	case "train":
		nn.TrainFromData(&net)
		nn.SaveModels(net)
	case "test":
		fmt.Println("Predicting from test dataset...")
		nn.LoadModels(&net)
		nn.PredictFromData(&net, "mnist_test")
	case "draw":
		fmt.Println("Predicting from drawing...")
		nn.LoadModels(&net)
		nn.PredictFromData(&net, "image")
	case "":
		fmt.Println("Mode not specified")
		return
	default:
		fmt.Println("Wrong mode")
		return
	}

	fmt.Printf("Time taken: %s", time.Since(start))
}
