package nn

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/gen2brain/raylib-go/raylib"
)

func main() {
	// d := Data{}
	// d.Create(28, 28)
	//
	// d.Fill()

	// fmt.Println(d.image)
	// fmt.Println(d.mat2)

	// draw.Run()

	// records := make(chan []string)
	// go openMNIST(records, 10, "train")
	// drawDigit(records)
	//
	// time.Sleep(time.Millisecond)
	// x, y, err := matSize(d.image)
	// if err != nil {
	// 	return
	// }
	// fmt.Println(x, y)

	// a := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	// b := [][]float64{{1, 2, 3}, {4, 5, 6}}
	// c, _ := matMul(a, b)
	// m, n, _ := matSize(a)
	// fmt.Println(c, m, n)
	// c, _ = matScalMut(c, 2)
	// fmt.Println(c)
}

// DrawDigit takes channel and draws records taken from the channel
func DrawDigit(records chan []string) {
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
				intensity, _ := strconv.Atoi(string(record[j+(i*28)+1]))
				color := rl.NewColor(uint8(intensity), uint8(intensity), uint8(intensity), 255)
				rl.DrawRectangle(int32(j)*scale, int32(i)*scale, scale, scale, color)
				rl.DrawText(record[0], 10, 5, 5*scale, rl.White)
			}
		}

		time.Sleep(500 * time.Millisecond)

		rl.EndDrawing()
	}

	rl.CloseWindow()
}

// OpenMNIST takes channel, number of numbers to read and mode (test/train) and puts records to the channel
func OpenMNIST(records chan []string, number int, mode string) {
	if mode != "train" && mode != "test" {
		return
	}

	path := fmt.Sprintf("data/mnist_%s.csv", mode)
	// path := "data/image.csv"
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}

	reader := csv.NewReader(file)

	for i := 0; i < number; i++ {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		records <- record
	}
	close(records)
}
