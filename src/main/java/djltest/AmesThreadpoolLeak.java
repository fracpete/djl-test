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
 * Process
 * 1. Models get trained via thread pool and then stored in static variables.
 * 2. Predictions on new data are made by submitting jobs to thread pool (using models/translators/predictors from static context).
 *
 * Massive memory leak.
 */
public class AmesThreadpoolLeak {

  public static Model[] models;

  public static Translator<ListFeatures, Float>[] translators;

  public static Predictor<ListFeatures, Float>[] predictors;

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

    // get translators
    translators = new Translator[numModels];
    for (int n = 0; n < numModels; n++) {
      System.out.println("translator: " + n);
      translators[n] = dataset.matchingTranslatorOptions().option(ListFeatures.class, Float.class);
    }

    // build models
    models = new Model[numModels];
    service = Executors.newFixedThreadPool(numModels);
    for (int n = 0; n < numModels; n++) {
      final int index = n;
      Callable<String> job = new Callable<>() {
	@Override
	public String call() throws Exception {
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
	  models[index] = model;
	  return null;
	}
      };
      service.submit(job);
    }
    waitForFinish(service);

    // instantiate predictors
    predictors = new Predictor[numModels];
    for (int n = 0; n < numModels; n++)
      predictors[n] = models[n].newPredictor(translators[n]);

    // predictions via threadpool
    Random rnd = new Random(1);
    for (int i = 0; i < 1000; i++) {
      service = Executors.newFixedThreadPool(numModels);
      for (int n = 0; n < numModels; n++) {
	final int index = n;
	Callable<String> job = new Callable<>() {
	  @Override
	  public String call() throws Exception {
	    ListFeatures features = new ListFeatures();
	    features.add("" + rnd.nextDouble()*1000);
	    features.add("" + rnd.nextDouble()*100);
	    features.add("" + rnd.nextInt(10));
	    Float pred = predictors[index].predict(features);
	    System.out.println(index + ": " + pred);
	    return null;
	  }
	};
	// memory leak happens even with delayed job submission!
	TimeUnit.MILLISECONDS.sleep(250);
	service.submit(job);
      }
      waitForFinish(service);
    }
  }
}
