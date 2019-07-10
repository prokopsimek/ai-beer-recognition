import * as tf from "@tensorflow/tfjs";

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
require("@tensorflow/tfjs-node");

// Train a simple model:
const model = tf.sequential();

model.add(
  tf.layers.dense({ units: 100, activation: "relu", inputShape: [10] })
);
model.add(tf.layers.dense({ units: 1, activation: "linear" }));
model.compile({ optimizer: "sgd", loss: "meanSquaredError" });

const xs = tf.randomNormal([1000, 10]);
const ys = tf.randomNormal([1000, 1]);

model.fit(xs, ys, {
  epochs: 1000,
  callbacks: {
    onEpochEnd: (epoch, log) => {
      console.log(`Epoch ${epoch}: loss = ${log.loss}`)
    }
  }
});


