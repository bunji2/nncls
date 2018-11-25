// Package nncls : NN による分類器のパッケージ
// Python で学習したモデルデータを使用
package nncls

import (
	"errors"
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Config : 設定パラメータ
type Config struct {
	ModelDir         string  // モデルデータのディレクトリ
	InputName        string  // 入力のプレースホルダの名前
	OutputName       string  // 出力のプレースホルダの名前
	NumInput         int     // 入力となる特徴量の個数
	LabelThreashould float32 // ラベル判定の閾値
}

var conf Config

// Init : 初期化
func Init(c Config) (err error) {
	conf = c
	if conf.ModelDir == "" {
		conf.ModelDir = "./model"
	}
	if conf.InputName == "" {
		conf.InputName = "INPUT"
	}
	if conf.OutputName == "" {
		conf.OutputName = "OUTPUT"
	}
	if conf.Threshould == 0.0 {
		conf.Threashould = 0.5
	}
	return
}

// Model : モデルデータを保持するデータ型
type Model struct {
	model *tf.SavedModel
}

// LoadModel : モデルデータの読み込み
func LoadModel() (r Model, err error) {
	if conf.ModelDir == "" {
		err = errors.New("conf.ModelDir is empty")
		return
	}
	var model *tf.SavedModel
	model, err = tf.LoadSavedModel(conf.ModelDir, []string{"serve"}, nil)
	r = Model{model: model}
	return
}

// Classify : 分類
// In:  testData --- 特徴データ
// Out: classID  --- クラスID
func (m Model) Classify(testData []float32) (classID int, err error) {
	var tensorTestData *tf.Tensor
	tensorTestData, err = tf.NewTensor([][]float32{testData})
	if err != nil {
		return
	}

	if conf.InputName == "" {
		err = errors.New("conf.InputName is empty")
		return
	}
	if conf.OutputName == "" {
		err = errors.New("conf.OutputName is empty")
		return
	}
	outputInput := m.model.Graph.Operation(conf.InputName)
	if outputInput == nil {
		err = errors.New("conf.InputName:" + conf.InputName + " is not found")
		m.DumpGraphOperations()
		return
	}
	outputOutput := m.model.Graph.Operation(conf.OutputName)
	if outputOutput == nil {
		err = errors.New("conf.OutputName:" + conf.OutputName + " is not found")
		m.DumpGraphOperations()
		return
	}

	feeds := map[tf.Output]*tf.Tensor{
		outputInput.Output(0): tensorTestData,
	}
	fetch := []tf.Output{
		outputOutput.Output(0),
	}

	var result []*tf.Tensor
	result, err = m.model.Session.Run(feeds, fetch, nil)

	if err != nil {
		return
	}

	valResult := result[0].Value().([][]float32)
	classID = maxIndex(valResult[0])

	return
}

// MultiLabelClassify : 分類
// In:  testData --- 特徴データ
// Out: classID  --- クラスID
func (m Model) MultiLabelClassify(testData []float32) (classIDs []int, err error) {
	var tensorTestData *tf.Tensor
	tensorTestData, err = tf.NewTensor([][]float32{testData})
	if err != nil {
		return
	}

	if conf.InputName == "" {
		err = errors.New("conf.InputName is empty")
		return
	}
	if conf.OutputName == "" {
		err = errors.New("conf.OutputName is empty")
		return
	}
	outputInput := m.model.Graph.Operation(conf.InputName)
	if outputInput == nil {
		err = errors.New("conf.InputName:" + conf.InputName + " is not found")
		m.DumpGraphOperations()
		return
	}
	outputOutput := m.model.Graph.Operation(conf.OutputName)
	if outputOutput == nil {
		err = errors.New("conf.OutputName:" + conf.OutputName + " is not found")
		m.DumpGraphOperations()
		return
	}

	feeds := map[tf.Output]*tf.Tensor{
		outputInput.Output(0): tensorTestData,
	}
	fetch := []tf.Output{
		outputOutput.Output(0),
	}

	var result []*tf.Tensor
	result, err = m.model.Session.Run(feeds, fetch, nil)

	if err != nil {
		return
	}

	valResult := result[0].Value().([][]float32)
	classIDs = pickupIndices(valResult[0])

	return
}

// DumpGraphOperations : モデルデータ内のグラフに含まれるオペレーション名を出力する関数
func (m Model) DumpGraphOperations() {
	fmt.Println("dumping names of Graph.Operations....")
	for i, operation := range m.model.Graph.Operations() {
		fmt.Println(i, operation.Name())
	}
}

// maxIndex : 配列の中で最大値となるインデックス番号を返す関数
func maxIndex(x []float32) (r int) {
	r = 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[r] {
			r = i
		}
	}
	return
}

// pickupIndices : 配列の中で閾値を超えるインデックス番号のリストを返す関数
func pickupIndices(x []float32) (r []int) {
	r = []int{}
	for i := 0; i < len(x); i++ {
		if x[i] > conf.Threashould {
			r = append(r, i)
		}
	}
	if len(r) < 1 {
		r = append(r, maxIndex(x))
	}
	return
}
