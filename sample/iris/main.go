package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/bunji2/nncls"
)

const (
	numClass = 3
	numInput = 4
)

var classNames = []string{
	"setosa",
	"versicolor",
	"virginica",
}

func main() {
	os.Exit(run())
}

func run() int {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s iris_test.csv\n", os.Args[0])
		return 1
	}

	testDataFile := os.Args[1]

	err := nncls.Init(nncls.Config{
		InputName:  "INPUT",
		OutputName: "OUTPUT/Softmax",
		ModelDir:   "./model",
		NumInput:   numInput,
	})
	model, err := nncls.LoadModel()
	if err != nil {
		fmt.Fprintln(os.Stderr, "run: Load:", err)
		return 2
	}

	var testX [][]float32
	var testY []int
	testX, testY, err = loadTestData(testDataFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "run: loadTestData:", err)
		return 3
	}

	fmt.Println("data,predict,teacher")

	md := nncls.NewMetricsData(numClass)

	var pred []float32
	var answer []float32
	var cid int
	for i := 0; i < len(testX); i++ {
		cid, err = model.Classify(testX[i])
		if err != nil {
			fmt.Fprintln(os.Stderr, "run: Classify:", err)
			break
		}
		pred, err = nncls.ToOneHot(cid, numClass)
		if err != nil {
			fmt.Fprintln(os.Stderr, "run: ToOneHot:", err)
			break
		}
		answer, err = nncls.ToOneHot(testY[i], numClass)
		if err != nil {
			fmt.Fprintln(os.Stderr, "run: ToOneHot:", err)
			break
		}
		for j := 0; j < numClass; j++ {
			md.Add(j, int(pred[j]), int(answer[j]))
		}
	}
	if err == nil {
		microPrecision, microRecall, microFMeasure, overallAccuracy := md.MicroMetrics()
		macroPrecision, macroRecall, macroFMeasure, averageAccuracy := md.MacroMetrics()
		fmt.Printf("micro\n")
		fmt.Printf("Precision:        %f\n", microPrecision)
		fmt.Printf("Recall:           %f\n", microRecall)
		fmt.Printf("F-Measure:        %f\n", microFMeasure)
		fmt.Printf("Overall Accuracy: %f\n", overallAccuracy)
		fmt.Printf("macro\n")
		fmt.Printf("Precision:        %f\n", macroPrecision)
		fmt.Printf("Recall:           %f\n", macroRecall)
		fmt.Printf("F-Measure:        %f\n", macroFMeasure)
		fmt.Printf("Average Accuracy: %f\n", averageAccuracy)
	}
	return 0
}

func loadTestData(testDataFile string) (testX [][]float32, testY []int, err error) {
	var fp *os.File
	fp, err = os.Open(testDataFile)
	if err != nil {
		return
	}
	defer fp.Close()

	reader := csv.NewReader(fp)
	reader.Comma = ','
	reader.LazyQuotes = true

	testX = [][]float32{}
	testY = []int{}

	var record []string
	var values []float32

	// skipping 1st line
	record, err = reader.Read()
	if err != nil {
		return
	}
	for {
		record, err = reader.Read()
		if err != nil {
			break
		}
		/*
			fmt.Println(record)
		*/
		values, err = toFloat32(record)
		if err != nil {
			break
		}
		testX = append(testX, values[0:4])
		testY = append(testY, int(values[4]))
	}
	if err == io.EOF {
		err = nil
	}

	return
}

func toFloat32(x []string) (r []float32, err error) {
	r = make([]float32, len(x))
	var tmp float64
	for i := 0; i < len(x); i++ {
		tmp, err = strconv.ParseFloat(x[i], 32)
		if err != nil {
			break
		}
		r[i] = float32(tmp)
	}
	return
}
