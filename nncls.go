// Package nncls : NN による分類器のパッケージ
// Python で学習したモデルデータを使用
package nncls

import (
	"fmt"

	"github.com/pkg/errors"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

// Config : 設定パラメータ
type Config struct {
	ModelDir   string  // モデルデータのディレクトリ
	InputName  string  // 入力のプレースホルダの名前
	OutputName string  // 出力のプレースホルダの名前
	NumInput   int     // 入力となる特徴量の個数
	Threshold  float32 // ラベル判定の閾値
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
	//if conf.Threshold == 0.0 {
	//	conf.Threshold = 0.5
	//}
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

// Predict : 分類
// In:  x --- 特徴データ
// Out: y --- クラスID(one_hot)
func (m Model) Predict(x []float32) (y []float32, err error) {
	var tensorTestData *tf.Tensor
	tensorTestData, err = tf.NewTensor([][]float32{x})
	if err != nil {
		err = errors.Wrap(err, "Predict: failed in tf.NewTensor")
		return
	}

	if conf.InputName == "" {
		err = errors.New("Predict: conf.InputName is empty")
		return
	}
	if conf.OutputName == "" {
		err = errors.New("Predict: conf.OutputName is empty")
		return
	}
	outputInput := m.model.Graph.Operation(conf.InputName)
	if outputInput == nil {
		err = errors.New("Predict: conf.InputName:" + conf.InputName + " is not found")
		m.DumpGraphOperations()
		return
	}
	outputOutput := m.model.Graph.Operation(conf.OutputName)
	if outputOutput == nil {
		err = errors.New("Predict: conf.OutputName:" + conf.OutputName + " is not found")
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
		err = errors.Wrap(err, "Predict: failed in m.model.Session.Run")
		return
	}

	valResult := result[0].Value().([][]float32)
	y = valResult[0]

	return
}

// Classify : 分類
// In:  testData --- 特徴データ
// Out: classID  --- クラスID
func (m Model) Classify(testData []float32) (classID int, err error) {
	var y []float32
	y, err = m.Predict(testData)
	if err != nil {
		err = errors.Wrap(err, "Classify: failed in m.Predict")
		return
	}
	classID = MaxIndex(y)

	return
}

// MultiLabelClassify : 分類
// In:  testData --- 特徴データ
// Out: classID  --- クラスID
func (m Model) MultiLabelClassify(testData []float32, f func([]float32) []int) (classIDs []int, err error) {

	if f == nil {
		err = errors.New("MultiLabelClassify: f is empty")
		return
	}

	var y []float32
	y, err = m.Predict(testData)
	if err != nil {
		err = errors.Wrap(err, "MultiLabelClassify: failed in m.Predict")
		return
	}

	classIDs = f(y)

	return
}

// DumpGraphOperations : モデルデータ内のグラフに含まれるオペレーション名を出力する関数
func (m Model) DumpGraphOperations() {
	fmt.Println("dumping names of Graph.Operations....")
	for i, operation := range m.model.Graph.Operations() {
		fmt.Println(i, operation.Name())
	}
}

// MaxIndex : 配列の中で最大値となるインデックス番号を返す関数
func MaxIndex(x []float32) (r int) {
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
		if x[i] > conf.Threshold {
			r = append(r, i)
		}
	}
	if len(r) < 1 {
		r = append(r, MaxIndex(x))
	}
	return
}

// RoundList : 配列の要素の中で閾値以下は0,閾値超は1となるリストを返す関数
func RoundList(x []float32) (r []int) {
	cnt := 0
	r = make([]int, len(x))
	for i := 0; i < len(x); i++ {
		if x[i] > conf.Threshold {
			r[i] = 1
			cnt++
		} else {
			r[i] = 0
		}
	}
	if cnt < 1 {
		r[MaxIndex(x)] = 1
	}
	return
}

// ToOneHot : スカラー値をクラス数 numClass の one-hot 形式の配列に変換
//            逆は MaxIndex。
func ToOneHot(classID, numClass int) (r []float32, err error) {
	if classID < 0 || classID >= numClass {
		err = fmt.Errorf("ToOneHot: classID(%d) is abnormal value against numClass(%d)", classID, numClass)
		return
	}
	r = make([]float32, numClass)
	r[classID] = float32(1)
	return
}
