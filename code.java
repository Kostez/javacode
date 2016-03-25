
import java.util.ArrayList;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author Kostez
 */
public class MLP {

    private int incline = 1;
    private ArrayList<int[]> samples;
    private ArrayList<int[]> answers;
    private int sizeNeuronsInput;
    private int sizeNeuronHidden = 8;           //поменять
    private int sizeNeuronsOuter;
    Layer hidden;
    Layer outer;
    double[][] tempWeightHidden;
    double[][] tempWeightOuter;
    double[] neuronsHidden;
    double[] neuronsOuter;

    double[] hidden_in;
    double[] outer_in;

    double[] errorHidden = new double[sizeNeuronHidden];
    double[] errorOuter = new double[sizeNeuronsInput];
    double[][] deltaWeightHidden = new double[sizeNeuronsInput + 1][sizeNeuronHidden];   //+1 - дельта-смещение
    double[][] deltaWeightOuter = new double[sizeNeuronHidden + 1][sizeNeuronsOuter];   //+1 - дельта-смещение

    int a = 1;      //скорость обучения
    int exitCondition = 10;

    double[] inputHiddenError = new double[sizeNeuronHidden];

    public MLP(ArrayList<int[]> samples, ArrayList<int[]> answers) {
        this.samples = samples;
        this.answers = answers;
        sizeNeuronsInput = samples.get(0).length;
        sizeNeuronsOuter = answers.get(0).length;
        hidden = new Layer(sizeNeuronsInput, sizeNeuronHidden);
        outer = new Layer(sizeNeuronHidden, sizeNeuronsOuter);
        neuronsHidden = hidden.getNeurons();
        neuronsOuter = outer.getNeurons();
        hidden_in = new double[sizeNeuronHidden];
        outer_in = new double[sizeNeuronsOuter];
    }

    public void study() {
        for (int i = 0; i < samples.size(); i++) {
            globalStudy(samples.get(i), answers.get(i));
        }
    }

    public void globalStudy(int[] sample, int[] answer) {
        //получаем сигналы нейронов скрытого слоя        
        
        tempWeightHidden = hidden.getNeuronsWeight();
        tempWeightOuter = outer.getNeuronsWeight();
        
        for (int j = 0; j < sizeNeuronHidden; j++) {
            hidden_in[j] = tempWeightHidden[sizeNeuronHidden][j];
            for (int i = 0; i < sizeNeuronsInput; i++) {
                hidden_in[j] += sample[i] * tempWeightHidden[i][j];
            }
            neuronsHidden[j] = activateFunction(hidden_in[j]);
        }
        //получаем сигналы нейронов исходящего слоя
        for (int k = 0; k < sizeNeuronsOuter; k++) {
            outer_in[k] = tempWeightOuter[sizeNeuronsOuter][k];
            for (int j = 0; j < sizeNeuronHidden; j++) {
                outer_in[k] += neuronsHidden[j] * tempWeightOuter[j][k];
            }
            neuronsOuter[k] = activateFunction(outer_in[k]);
        }

        //условие прекращения обучения
        
        if (exitCondition > calcSCO(answer, neuronsOuter)) {
            
        }
        
        
        errorOuter = calcErrorOut(answer, neuronsOuter, outer_in);
        deltaWeightOuter = calcDeltaWeight(errorOuter, neuronsOuter);
        inputHiddenError = culcInputHiddenError(errorOuter, tempWeightOuter);
        errorHidden = calcErrorHidden(inputHiddenError, hidden_in);
        deltaWeightHidden = calcDeltaWeight(errorHidden, neuronsHidden);

        //8 шаг: именение весов
        outer.setNeuronsWeight(changeWeight(tempWeightOuter, deltaWeightOuter));
        hidden.setNeuronsWeight(changeWeight(tempWeightHidden, deltaWeightHidden));
    }
    
    private double calcSCO(int[] answer, double[] neuronsOuter){
        double sCO = 0;
        for (int i = 0; i < answer.length; i++) {
            sCO = (Math.pow(answer[i] + neuronsOuter[i], 2)/answer.length);
        }
        return sCO;
    }
    
    private double[] calcErrorHidden(double[] inputHiddenError, double[] hidden_in) {
        double[] temperrorHidden = new double[inputHiddenError.length];
        for (int j = 0; j < hidden_in.length; j++) {
            temperrorHidden[j] = inputHiddenError[j] * activateFunctionDerivative(hidden_in[j]);
        }
        return temperrorHidden;
    }

    private double[] culcInputHiddenError(double[] errors, double[][] outerWeight) {
        double[] tempInputHiddenError = new double[errors.length];
        for (int j = 0; j < outerWeight.length - 1; j++) {
            tempInputHiddenError[j] = 0;
            for (int k = 0; k < errors.length; k++) {
                tempInputHiddenError[j] += errors[k] * outerWeight[j][k];
            }
        }
        return tempInputHiddenError;
    }

    private double[] calcErrorOut(int[] answer, double[] neuronsOuter, double[] outer_in) {
        double[] tempErrorOuter = new double[neuronsOuter.length];
        for (int k = 0; k < outer_in.length; k++) {
            tempErrorOuter[k] = (answer[k] - neuronsOuter[k]) * activateFunctionDerivative(outer_in[k]);
        }
        return tempErrorOuter;
    }

    private double[][] calcDeltaWeight(double[] error, double[] neurons) {                   ///
        double[][] tempDeltaWeight = new double[neurons.length + 1][error.length];
        for (int i = 0; i < neurons.length; i++) {
            for (int j = 0; j < error.length; j++) {
                tempDeltaWeight[i][j] = a * error[j] * neurons[i];
            }
            for (int j = 0; j < error.length; j++) {
                tempDeltaWeight[neurons.length][j] = a * error[j];                   //величина корректировки смещения
            }
        }
        return tempDeltaWeight;
    }

    public double activateFunction(double x) {
        return 1 / (1 - Math.exp(-incline * x));
    }

    public double activateFunctionDerivative(double x) {
        double activateFunctionX = activateFunction(x);
        return activateFunctionX * (1 - activateFunctionX);
    }

    private double[][] changeWeight(double[][] tempWeight, double[][] deltaWeight) {
        double[][] tempChangeWeight = new double[tempWeight.length][tempWeight[0].length];
        for (int j = 0; j < tempChangeWeight.length; j++) {
            for (int k = 0; k < tempChangeWeight[j].length; k++) {
                tempChangeWeight[j][k] += deltaWeight[j][k];
            }
        }
        return tempChangeWeight;
    }
}
