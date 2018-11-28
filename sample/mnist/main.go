package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/bunji2/nncls"
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
		NumInput:   784,
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

	m := nncls.NewMetricsData(10)
	okCount := 0

	var y, pred, answer []float32
	var classID int
	for i := 0; i < len(testX); i++ {
		//classID, err = model.Classify(testX[i])
		y, err = model.Predict(testX[i])
		if err != nil {
			fmt.Fprintln(os.Stderr, "run: Classify:", err)
			break
		}
		//fmt.Println("pred =", y)
		answer, err = nncls.ToOneHot(testY[i], 10)
		//fmt.Println("answer =", answer)

		classID = nncls.MaxIndex(y)

		pred, err = nncls.ToOneHot(classID, 10)
		if err != nil {
			break
		}
		for j := 0; j < 10; j++ {
			m.Add(j, int(pred[j]), int(answer[j]))
		}

		fmt.Printf("%d,%d,", classID, testY[i])
		if classID == testY[i] {
			fmt.Println("OK")
			okCount++
		} else {
			fmt.Println("NG")
		}
	}
	if err != nil {
		return 4
	}
	fmt.Println("classID,Precision,Recall,Accuracy")
	for j := 0; j < 10; j++ {
		fmt.Printf("%d,%f,%f,%f\n", j, m.Precision(j), m.Recall(j), m.Accuracy(j))
	}
	p, r, f, a := m.MicroMetrics()
	fmt.Println("#,Precision,Recall,F-measure,Accuracy")
	fmt.Printf("micro,%f,%f,%f,%f\n", p, r, f, a)
	p, r, f, a = m.MicroMetrics()
	fmt.Printf("macro,%f,%f,%f,%f\n", p, r, f, a)
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
		testX = append(testX, values[0:784])
		testY = append(testY, int(values[784]))
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
