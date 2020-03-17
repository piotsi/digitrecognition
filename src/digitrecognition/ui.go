package main

import (
	"fmt"

	rl "github.com/gen2brain/raylib-go/raylib"
)

var (
	mousePos     rl.Vector2
	mousePosLast rl.Vector2
)

const (
	windowHeight = 600
	windowWidth  = 400
)

// Draws is draw
func main() {
	rl.InitWindow(windowWidth, windowHeight, "GoDraw")
	rl.SetTargetFPS(120)

	// Load clean canvas
	canvas := rl.LoadRenderTexture(windowWidth, windowHeight)
	rl.BeginTextureMode(canvas)
	rl.ClearBackground(rl.White)
	rl.EndTextureMode()

	var brushSize float32 = 10
	chosenColor := rl.Black
	toClear := false

	for !rl.WindowShouldClose() {
		// --------
		// Updating
		// --------
		mousePos = rl.GetMousePosition()

		// Save texture to png file
		if rl.IsKeyPressed(rl.KeyS) {
			image := rl.GetTextureData(canvas.Texture)
			rl.ImageFlipVertical(image)
			rl.ExportImage(*image, "images/mypainting.png")
			rl.UnloadImage(image)

			// Clear canvas
			toClear = true
		}

		// Run neural network to recognize digit
		if rl.IsKeyPressed(rl.KeyR) {
			NN()
		}

		// -------
		// Drawing
		// -------
		rl.BeginDrawing()
		rl.ClearBackground(rl.White)

		// Paint on a texture
		if rl.IsMouseButtonPressed(rl.MouseLeftButton) || rl.GetGestureDetected() == rl.GestureDrag {
			rl.BeginTextureMode(canvas)
			// rl.DrawCircle(int32(mousePos.X), int32(mousePos.Y), brushSize, chosenColor)
			rl.DrawLineEx(mousePos, mousePosLast, brushSize, chosenColor)
			rl.EndTextureMode()
		}

		// Draw texture,
		// height is flipped because Raylib uses top-left corner starting point
		// but OpenGl uses bottom-left approach
		rl.DrawTextureRec(canvas.Texture, rl.Rectangle{X: 0, Y: 0, Width: windowWidth, Height: -windowHeight}, rl.Vector2{X: 0, Y: 0}, rl.White)

		// Draw brush "cursor" and outline
		rl.DrawCircle(int32(mousePos.X), int32(mousePos.Y), brushSize, chosenColor)
		rl.DrawCircleLines(int32(mousePos.X), int32(mousePos.Y), brushSize, rl.Black)
		rl.DrawText(fmt.Sprintf("%.f %.f", mousePos.X, mousePos.Y), 20, 20, 20, rl.Black)

		// Clear the canvas
		if rl.IsKeyPressed(rl.KeyC) || toClear {
			rl.BeginTextureMode(canvas)
			rl.ClearBackground(rl.White)
			rl.EndTextureMode()
			toClear = false
		}

		mousePosLast = mousePos

		rl.EndDrawing()
	}

	// Unload texture from GPU memory
	rl.UnloadRenderTexture(canvas)
	rl.CloseWindow()
}
