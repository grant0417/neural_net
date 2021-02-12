use nalgebra::{DMatrix, Dynamic, MatrixXx1, U1};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

struct MnistDataset {
    records: Vec<MnistRecord>,
}

struct MnistRecord {
    target: usize,
    targets: Vec<f32>,
    inputs: Vec<f32>,
}

fn load_mnist_data<P: AsRef<Path>>(path: P) -> Result<MnistDataset, String> {
    println!("Loading data from {}...", path.as_ref().display());
    let test_file = BufReader::new(File::open(path.as_ref()).map_err(|e| e.to_string())?);
    let mut csv_reader = csv::Reader::from_reader(test_file);

    let mut records = Vec::new();
    for result in csv_reader.records() {
        let record = result.map_err(|e| e.to_string())?;

        let inputs: Vec<f32> = (0..784)
            .into_iter()
            .map(|i| record[i + 1].parse::<f32>().unwrap() / 255.0)
            .collect();

        let target = record[0].parse::<usize>().map_err(|e| e.to_string())?;
        let mut targets = vec![0.; 10];
        targets[target] = 1.;

        records.push(MnistRecord {
            target,
            targets,
            inputs,
        });
    }
    Ok(MnistDataset { records })
}

#[derive(Debug, Serialize, Deserialize)]
struct Network {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    hidden_weights: DMatrix<f32>,
    output_weights: DMatrix<f32>,
    learning_rate: f32,
}

impl Network {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f32) -> Self {
        let mut rng = thread_rng();
        Self {
            input_size,
            hidden_size,
            output_size,
            learning_rate,
            hidden_weights: DMatrix::from_distribution(
                hidden_size,
                input_size,
                &rand::distributions::Uniform::new(
                    -1. / (input_size as f32).sqrt(),
                    1. / (input_size as f32).sqrt(),
                ),
                &mut rng,
            ),
            output_weights: DMatrix::from_distribution(
                output_size,
                hidden_size,
                &rand::distributions::Uniform::new(
                    -1. / (hidden_size as f32).sqrt(),
                    1. / (hidden_size as f32).sqrt(),
                ),
                &mut rng,
            ),
        }
    }

    fn predict(&self, input_data: &[f32]) -> Result<MatrixXx1<f32>, String> {
        if input_data.len() != self.input_size {
            return Err(String::from("input_data length != inputs"));
        }

        let inputs = MatrixXx1::from_column_slice(input_data);

        let sigmoid = |x: f32| 1. / (1. + (-1. * x).exp());

        let mut hidden_outputs = &self.hidden_weights * inputs;
        hidden_outputs.apply(sigmoid);
        let mut final_outputs = &self.output_weights * hidden_outputs;
        final_outputs.apply(sigmoid);

        Ok(final_outputs.reshape_generic(Dynamic::new(self.output_size), U1))
    }

    fn train(&mut self, input_data: &[f32], target_data: &[f32]) -> Result<(), String> {
        if input_data.len() != self.input_size {
            return Err(String::from("input_data length != inputs"));
        }
        if target_data.len() != self.output_size {
            return Err(String::from("targer_data length != outputs"));
        }

        let sigmoid = |x: f32| 1. / (1. + (-1. * x).exp());
        let sigmoid_dx = |x: f32| x * (1. - x);

        // Forward propagation
        let inputs = MatrixXx1::from_column_slice(input_data);
        let mut hidden_outputs = &self.hidden_weights * &inputs;
        hidden_outputs.apply(sigmoid);
        let mut final_outputs = &self.output_weights * &hidden_outputs;
        final_outputs.apply(sigmoid);

        // Errors
        let targets = MatrixXx1::from_column_slice(target_data);
        let output_errors = targets - &final_outputs;
        let hidden_errors = &self.output_weights.transpose() * &output_errors;

        // Backpropagation
        self.output_weights = &self.output_weights
            + (output_errors.component_mul(&final_outputs.map(sigmoid_dx))
                * &hidden_outputs.transpose())
                .scale(self.learning_rate);

        self.hidden_weights = &self.hidden_weights
            + (hidden_errors.component_mul(&hidden_outputs.map(sigmoid_dx)) * inputs.transpose())
                .scale(self.learning_rate);

        Ok(())
    }

    fn mnist_train(&mut self, epochs: usize) -> Result<(), String> {
        let dataset = load_mnist_data("data/mnist_train.csv")?.records;

        let start = std::time::Instant::now();

        for epoch in 0..epochs {
            println!("Epoch {}/{}...", epoch + 1, epochs);
            for (
                i,
                MnistRecord {
                    targets, inputs, ..
                },
            ) in dataset.iter().enumerate()
            {
                self.train(&*inputs, &*targets)?;
                if i % (dataset.len() / 10) == 0 {
                    println!("{}%", i * 100 / dataset.len());
                }
            }
        }

        let elapsed = start.elapsed();

        println!(
            "Time taken to train: {}m {}s",
            elapsed.as_secs() / 60,
            (elapsed.as_millis() as f32 / 1000.) % 60.,
        );

        Ok(())
    }

    fn mnist_predict(&self) -> Result<(), String> {
        let dataset = load_mnist_data("data/mnist_test.csv")?.records;

        let start = std::time::Instant::now();

        let mut correct = 0;
        for MnistRecord { target, inputs, .. } in dataset.iter() {
            let (best, _best_val) = self.predict(inputs)?.argmax();
            if *target == best {
                correct += 1;
            }
        }

        let elapsed = start.elapsed();

        println!(
            "Time taken to predict: {}m {}s",
            elapsed.as_secs() / 60,
            (elapsed.as_millis() as f32 / 1000.) % 60.,
        );
        println!(
            "Accuracy: {}% ({}/{})",
            (correct as f32 / dataset.len() as f32) * 100.,
            correct,
            dataset.len()
        );
        Ok(())
    }

    fn save_file<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        println!("Saving to file...");
        let network_file = BufWriter::new(File::create(path.as_ref()).map_err(|e| e.to_string())?);
        serde_json::to_writer_pretty(network_file, &self).unwrap();
        Ok(())
    }

    fn load_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        println!("Loading from file...");
        let network_file = BufReader::new(File::open(path.as_ref()).map_err(|e| e.to_string())?);
        let net: Self = serde_json::from_reader(network_file).map_err(|e| e.to_string())?;
        println!("Loaded {}-{}-{}", net.input_size, net.hidden_size, net.output_size);
        Ok(net)
    }
}

fn main() -> Result<(), String> {
    let mut n = Network::new(784, 200, 10, 0.05);
    n.mnist_train(5)?;
    n.save_file("data/network.model")?;
    let model = Network::load_file("data/network.model")?;
    model.mnist_predict()?;
    Ok(())
}
