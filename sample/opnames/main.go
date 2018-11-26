package main

// モデルデータのグラフに含まれる「オペレーション」の名前をダンプする

import (
	"fmt"
	"os"

	"github.com/bunji2/nncls"
)

func main() {
	os.Exit(run())
}

func run() int {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s modelDir\n", os.Args[0])
		return 1
	}

	modelDir := os.Args[1]

	err := nncls.Init(nncls.Config{ModelDir: modelDir})
	model, err := nncls.LoadModel()
	if err != nil {
		fmt.Fprintln(os.Stderr, "run: LoadModel:", err)
		return 2
	}

	model.DumpGraphOperations()
	return 0
}
