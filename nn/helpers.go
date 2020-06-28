package nn

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"time"

	imagecolor "image/color"

	"github.com/gen2brain/raylib-go/raylib"
	imgcat "github.com/martinlindhe/imgcat/lib"
)

// DrawDigitsWindow takes channel and draws records taken from the channel in the window
func DrawDigitsWindow(records chan []float64) {
	rl.SetConfigFlags(rl.FlagVsyncHint)
	var scale int32 = 10
	rl.InitWindow(28*scale, 28*scale, "image")

	rl.SetTargetFPS(60)

	for !rl.WindowShouldClose() {
		rl.BeginDrawing()

		rl.ClearBackground(rl.RayWhite)

		record, ok := <-records

		if !ok {
			time.Sleep(2000 * time.Millisecond)
			return
		}

		for i := 0; i < 28; i++ {
			for j := 0; j < 28; j++ {
				intensity := record[j+(i*28)+1]
				color := rl.NewColor(uint8(intensity), uint8(intensity), uint8(intensity), 255)
				rl.DrawRectangle(int32(j)*scale, int32(i)*scale, scale, scale, color)
				// rl.DrawText(float64(strconv.Atoi(record[0])), 10, 5, 5*scale, rl.White)
			}
		}

		time.Sleep(500 * time.Millisecond)

		rl.EndDrawing()
	}

	rl.CloseWindow()
}

// DrawDigitTerminal takes slice of strings and draws digit in the terminal
func DrawDigitTerminal(record []float64) {
	img := image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{27, 27}})
	for i := 0; i < 28; i++ {
		for j := 0; j < 28; j++ {
			intensity := 255 * (record[j+(i*28)+1])
			img.Set(j, i, imagecolor.RGBA{uint8(intensity), uint8(intensity), uint8(intensity), 0xff})
		}
	}

	imgcat.CatImage(img, os.Stdout)
}

// OpenCSV takes channel, number of numbers to read and mode (test/train) and puts records to the channel
func OpenCSV(records chan []float64, number int, mode string) {
	if mode != "mnist_train" && mode != "mnist_test" && mode != "image" {
		return
	}

	path := fmt.Sprintf("data/%s.csv", mode)
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}

	reader := csv.NewReader(file)

	for i := 0; i < number; i++ {
		recordS, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		record := make([]float64, len(recordS))
		for j, l := range recordS {
			temp, _ := strconv.ParseFloat(l, 64)
			record[j] = (temp / 255 * 0.99) + 0.01
			if j == 0 {
				record[0] = temp
			}
		}

		records <- record
	}
	// close(records)
	file.Close()
}

// Classify returns index of highest probability of the output
func Classify(output Mat) (int, int) {
	max := 0.0
	hp := 10
	for i, v := range output {
		if v[0] > max {
			max = v[0]
			hp = i
		}
	}
	return hp, int(max * 100)
}

// SaveModels saves hiddenWeights and outputWeights to a .model file
func SaveModels(net Network) {
	hwmodel, err := os.Create("data/weightshidden.json")
	defer hwmodel.Close()
	if err == nil {
		file, _ := json.Marshal(net.hiddenWeights)
		_ = ioutil.WriteFile("data/weightshidden.json", file, 0644)
	}

	owmodel, err := os.Create("data/weightsoutput.json")
	defer owmodel.Close()
	if err == nil {
		file, _ := json.Marshal(net.outputWeights)
		_ = ioutil.WriteFile("data/weightsoutput.json", file, 0644)
	}
}

// LoadModels loads hiddenWeights and outputWeights from a .model file
func LoadModels(net *Network) {
	hwmodel, err := ioutil.ReadFile("data/weightshidden.json")
	if err == nil {
		json.Unmarshal(hwmodel, &net.hiddenWeights)
	}

	owmodel, err := ioutil.ReadFile("data/weightsoutput.json")
	if err == nil {
		json.Unmarshal(owmodel, &net.outputWeights)
	}
	return
}
