package djltest;

import ai.djl.Model;
import ai.djl.basicdataset.tabular.AmesRandomAccess;
import ai.djl.basicdataset.tabular.ListFeatures;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.TabNetRegressionLoss;
import ai.djl.translate.Translator;
import ai.djl.zero.Performance;

import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Runs 4 separate pipelines via a thread pool:
 * train model, loop for making predictions on new data
 *
 * No memory leak.
 */
public class AmesThreadpoolNoLeak {

  public static void waitForFinish(ExecutorService service) {
    service.shutdown();
    while (!service.isTerminated()) {
      try {
	service.awaitTermination(100, TimeUnit.MILLISECONDS);
      }
      catch (InterruptedException e) {
	// ignored
      }
      catch (Exception e) {
	e.printStackTrace();
      }
    }
  }

  public static void main(String[] args) throws Exception {
    ExecutorService service;
    int numModels = 4;

    // load data
    AmesRandomAccess dataset = AmesRandomAccess.builder()
				 .setSampling(32, true)
				 .addNumericFeature("lotarea")
				 .addNumericFeature("miscval")
				 .addNumericFeature("overallqual")
				 .addNumericLabel("saleprice")
				 .build();
    Dataset[] splitDataset = dataset.randomSplit(8, 2);
    Dataset trainDataset = splitDataset[0];
    Dataset validateDataset = splitDataset[1];

    // separate thread for each model build/prediction pipeline
    service = Executors.newFixedThreadPool(numModels);
    for (int n = 0; n < numModels; n++) {
      final int index = n;
      Callable<String> job = new Callable<>() {
	@Override
	public String call() throws Exception {
	  // translator
	  System.out.println("translator: " + index);
	  Translator<ListFeatures, Float> translator = dataset.matchingTranslatorOptions().option(ListFeatures.class, Float.class);

	  // train
	  System.out.println("train: " + index);
	  Block block = TabularRegression.createBlock(Performance.FAST, dataset.getFeatureSize(), dataset.getLabelSize());
	  Model model = Model.newInstance("tabular");
	  model.setBlock(block);

	  TrainingConfig trainingConfig =
	    new DefaultTrainingConfig(new TabNetRegressionLoss())
	      .addTrainingListeners(TrainingListener.Defaults.basic());

	  try (Trainer trainer = model.newTrainer(trainingConfig)) {
	    trainer.initialize(new Shape(1, dataset.getFeatureSize()));
	    EasyTrain.fit(trainer, 5, trainDataset, validateDataset);
	  }

	  // predict
	  System.out.println("predict: " + index);
	  Random rnd = new Random(1);
	  Predictor<ListFeatures, Float> predictor = model.newPredictor(translator);
	  for (int i = 0; i < 10000; i++) {
	    ListFeatures features = new ListFeatures();
	    features.add("" + rnd.nextDouble()*1000);
	    features.add("" + rnd.nextDouble()*100);
	    features.add("" + rnd.nextInt(10));
	    Float pred = predictor.predict(features);
	    System.out.println(index + ": " + pred);
	  }

	  return null;
	}
      };
      service.submit(job);
    }
    waitForFinish(service);
  }
}
