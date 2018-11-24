package main

import (
	"fmt"
	"os"

	"github.com/bunji2/nncls"
	gm "github.com/petar/GoMNIST"
)

func main() {
	os.Exit(run())
}

func run() int {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s ./data_set\n", os.Args[0])
		return 1
	}

	dataDir := os.Args[1]

	err := nncls.Init(nncls.Config{})
	model, err := nncls.LoadModel()
	if err != nil {
		fmt.Fprintln(os.Stderr, "run: Load:", err)
		return 2
	}

	var testData *gm.Set

	testData, _, err = gm.Load(dataDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "run: loadTestData:", err)
		return 3
	}

	fmt.Println(toFloat32(testData.Images[0]))

	fmt.Println("data,predict,teacher")

	okCount := 0

	var classID int
	var testX []float32
	var testY int
	for i := 0; i < len(testData.Images); i++ {
		testX, err = toFloat32(testData.Images[i])
		if err != nil {
			fmt.Fprintln(os.Stderr, "run: toFloat32:", err)
			break
		}
		testY = int(testData.Labels[i])
		classID, err = model.Classify(testX)
		if err != nil {
			fmt.Fprintln(os.Stderr, "run: Classify:", err)
			break
		}
		fmt.Printf("%d,%d,", classID, testY)
		if classID == testY {
			fmt.Println("OK")
			okCount++
		} else {
			fmt.Println("NG")
		}
	}
	if err != nil {
		return 4
	}
	fmt.Printf("accuracy : %f\n", float32(okCount)/float32(len(testData.Images)))
	return 0
}

func toFloat32(x []byte) (r []float32, err error) {
	r = make([]float32, len(x))
	for i := 0; i < len(x); i++ {
		r[i] = float32(x[i]) / 256.0
	}
	return
}
