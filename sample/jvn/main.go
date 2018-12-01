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
	numClass = 13
	numInput = 669
)

func main() {
	os.Exit(run())
}

func run() int {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s test.csv\n", os.Args[0])
		return 1
	}

	testDataFile := os.Args[1]

	err := nncls.Init(nncls.Config{
		InputName:  "INPUT",
		OutputName: "OUTPUT",
		ModelDir:   "./model",
		NumInput:   numInput,
		Threshold:  0.0,
	})
	model, err := nncls.LoadModel()
	if err != nil {
		fmt.Fprintln(os.Stderr, "run: Load:", err)
		return 2
	}

	var testX [][]float32
	var testY [][]int

	testX, testY, err = loadTestData(testDataFile)
	if err != nil {
		fmt.Fprintln(os.Stderr, "run: loadTestData:", err)
		return 3
	}

	fmt.Println("data,predict,teacher")

	md := nncls.NewMetricsData(numClass)

	var classIDs []int
	for i := 0; i < len(testX); i++ {
		classIDs, err = model.MultiLabelClassify(testX[i], nncls.RoundList)
		if err != nil {
			fmt.Fprintln(os.Stderr, "run: MultiLabelClassify:", err)
			break
		}
		fmt.Printf("%v\n%v\n", classIDs, testY[i])
		if isEqual(classIDs, testY[i]) {
			fmt.Println("===>OK")
			//okCount++
		} else {
			fmt.Println("===>NG")
		}
		err = md.AddLabels(classIDs, testY[i])
		if err != nil {
			fmt.Fprintln(os.Stderr, "run: AddLabels:", err)
			break
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

func isEqual(x, y []int) (r bool) {
	if len(x) != len(y) {
		return
	}
	for i := 0; i < len(x); i++ {
		if x[i] != y[i] {
			return
		}
	}
	r = true
	return
}

func loadTestData(testDataFile string) (testX [][]float32, testY [][]int, err error) {
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
	testY = [][]int{}

	var record []string
	var fValues []float32
	var iValues []int

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
		fValues, err = toFloat32(record[0:numInput])
		if err != nil {
			break
		}
		testX = append(testX, fValues)
		iValues, err = toInt(record[numInput:])
		if err != nil {
			break
		}
		testY = append(testY, iValues)
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

func toInt(x []string) (r []int, err error) {
	r = make([]int, len(x))
	var tmp float64
	for i := 0; i < len(x); i++ {
		tmp, err = strconv.ParseFloat(x[i], 32)
		if err != nil {
			break
		}
		r[i] = int(tmp)
	}
	return
}
