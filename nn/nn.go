package nn

import (
	"fmt"
	"strings"
	"time"

	draw "github.com/piotsik/draw"
)

// Network is three layer feedforward network (perceptron)
type Network struct {
	inputNeurons  int
	hiddenNeurons int
	outputNeurons int
	hiddenWeights Mat
	outputWeights Mat
	learingRate   float64
}

// Construct constructs new network
func Construct(input, hidden, output int, rate float64) (net Network) {
	net = Network{
		inputNeurons:  input,
		hiddenNeurons: hidden,
		outputNeurons: output,
		learingRate:   rate,
	}

	net.hiddenWeights = CreateRandom(net.hiddenNeurons, net.inputNeurons, float64(net.inputNeurons))
	net.outputWeights = CreateRandom(net.outputNeurons, net.hiddenNeurons, float64(net.hiddenNeurons))

	return
}

// Predict predicts values
func (net Network) Predict(record []float64) Mat {
	hiddenInputs := Dot(net.hiddenWeights, SliceToMat(record[1:]))
	hiddenOutputs := ApplyFunc(hiddenInputs, Sigmoid)
	finalInputs := Dot(net.outputWeights, hiddenOutputs)
	finalOutputs := ApplyFunc(finalInputs, Sigmoid)

	return finalOutputs
}

// Train trains
func (net *Network) Train(record []float64) {
	hiddenInputs := Dot(net.hiddenWeights, SliceToMat(record[1:]))
	hiddenOutputs := ApplyFunc(hiddenInputs, Sigmoid)
	finalInputs := Dot(net.outputWeights, hiddenOutputs)
	finalOutputs := ApplyFunc(finalInputs, Sigmoid)

	targetsSlice := make([]float64, 10)
	for i := range targetsSlice {
		targetsSlice[i] = 0.01
	}
	targetsSlice[int(record[0])] = 0.99
	targets := SliceToMat(targetsSlice)

	errorOutputs := Subtract(targets, finalOutputs)
	errorsHidden := Dot(Transpose(net.outputWeights), errorOutputs)

	net.outputWeights = Add(net.outputWeights,
		ScalarMul(
			Dot(Mul(errorOutputs, ApplyFunc(finalOutputs, SigmoidPrime)), Transpose(hiddenOutputs)),
			net.learingRate))
	net.hiddenWeights = Add(net.hiddenWeights,
		ScalarMul(
			Dot(Mul(errorsHidden, ApplyFunc(hiddenOutputs, SigmoidPrime)), Transpose(SliceToMat(record[1:]))),
			net.learingRate))
}

// TrainFromData trains the network from the provided data
func TrainFromData(net *Network) {
	epochs := 5
	toTrain := 60000

	for epoch := 0; epoch < epochs; epoch++ {
		records := make(chan []float64)
		go OpenCSV(records, toTrain, "mnist_train")
		i := 1
		for {
			record, ok := <-records
			if !ok {
				break
			}
			net.Train(record)
			fmt.Printf("Training%-3s Epoch: %d/%d Record: %5d/%d   %3d%%\r", strings.Repeat(".", i%4), epoch+1, epochs, i, toTrain, 100*(epoch*toTrain+i)/(epochs*toTrain))
			i++
		}
		close(records)
	}
}

// PredictFromData predicts the input
func PredictFromData(net *Network, mode string) {
	if mode == "image" {
		records := make(chan []float64)

		go func() {
			for {
				record, _ := <-records

				a, b := Classify(net.Predict(record))
				draw.Text = fmt.Sprintf("looks to me like %d, i'm %d%% sure", a, b)
				time.Sleep(1 * time.Second)
			}
		}()
		go func() {
			for {
				PNGtoCSV()
				OpenCSV(records, 100, mode)
			}
		}()
		draw.Run(true)
		return
	}
	records := make(chan []float64)
	go OpenCSV(records, 100, mode)

	for {
		record, ok := <-records
		if !ok {
			break
		}

		DrawDigitTerminal(record)
		a, b := Classify(net.Predict(record))
		fmt.Printf("looks to me like %d, i'm %d%% sure\n", a, b)
	}

	return
}
