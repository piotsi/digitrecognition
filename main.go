package main

import (
	nn "digitrecognition/nn"
	"time"
)

func main() {
	nn.PNGtoCSV()
	records := make(chan []string)
	go nn.OpenMNIST(records, 10, "train")
	nn.DrawDigit(records)

	time.Sleep(time.Millisecond)
}
