package nn

import (
	"encoding/csv"
	"image"
	_ "image/png" // a
	"log"
	"os"
	"strconv"
)

// PNGtoCSV converts image to csv file
func PNGtoCSV() {
	// Open and decode image
	filePNG, err := os.Open("images/image.png")
	if err != nil {
		log.Fatal(err)
	}
	defer filePNG.Close()

	src, _, err := image.Decode(filePNG)
	if err != nil {
		log.Fatal(err)
	}

	// Create and write decoded image to a csv file
	fileCSV, err := os.Create("data/image.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer fileCSV.Close()

	writer := csv.NewWriter(fileCSV)
	defer writer.Flush()

	bounds := src.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	data := []string{"unrcgnzd"}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			gray, _, _, _ := src.At(x, y).RGBA()
			data = append(data, strconv.Itoa(int(gray)/257))
		}
	}

	err = writer.Write(data)
	if err != nil {
		log.Fatal(err)
	}

}
