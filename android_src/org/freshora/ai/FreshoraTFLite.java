package org.freshora.ai;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

/**
 * Android-native LiteRT Interpreter bridge used by Freshora's Python predictor.
 *
 * The LiteRT 1.4.2 Interpreter artifact retains the org.tensorflow.lite Java
 * API and supports the FULLY_CONNECTED v12 operator used by this model.
 */
public final class FreshoraTFLite {
    private static final int EXPECTED_INPUT_BYTES = 1 * 224 * 224 * 3 * 4;

    private final Interpreter interpreter;
    private final int outputSize;
    private final int inputByteCount;
    private final ByteBuffer inputBuffer;
    private final float[][] outputBuffer;

    public FreshoraTFLite(String modelPath) throws Exception {
        if (modelPath == null || modelPath.trim().isEmpty()) {
            throw new IllegalArgumentException("Model path is empty");
        }

        File modelFile = new File(modelPath);
        if (!modelFile.isFile() || !modelFile.canRead()) {
            throw new IllegalArgumentException(
                "TFLite model is missing or unreadable: " + modelPath
            );
        }

        Interpreter.Options options = new Interpreter.Options();
        // A modest thread count avoids excessive CPU/RAM use on older phones.
        options.setNumThreads(2);
        interpreter = new Interpreter(modelFile, options);

        Tensor input = interpreter.getInputTensor(0);
        Tensor output = interpreter.getOutputTensor(0);
        int[] inputShape = input.shape();
        int[] outputShape = output.shape();

        if (input.dataType() != DataType.FLOAT32) {
            interpreter.close();
            throw new IllegalArgumentException(
                "Model input must be FLOAT32; found " + input.dataType()
            );
        }
        if (output.dataType() != DataType.FLOAT32) {
            interpreter.close();
            throw new IllegalArgumentException(
                "Model output must be FLOAT32; found " + output.dataType()
            );
        }
        if (inputShape.length != 4 || inputShape[0] != 1 ||
            inputShape[1] != 224 || inputShape[2] != 224 ||
            inputShape[3] != 3) {
            interpreter.close();
            throw new IllegalArgumentException(
                "Expected input shape [1,224,224,3]; found " +
                shapeToString(inputShape)
            );
        }
        if (outputShape.length != 2 || outputShape[0] != 1 ||
            outputShape[1] <= 0) {
            interpreter.close();
            throw new IllegalArgumentException(
                "Expected output shape [1,classes]; found " +
                shapeToString(outputShape)
            );
        }

        inputByteCount = input.numBytes();
        outputSize = outputShape[1];
        if (inputByteCount != EXPECTED_INPUT_BYTES) {
            interpreter.close();
            throw new IllegalArgumentException(
                "Unexpected input tensor byte count: " + inputByteCount
            );
        }

        // Keep one native input buffer and one output array for all scans.
        // Reusing them reduces native-memory churn on low-RAM phones.
        inputBuffer = ByteBuffer.allocateDirect(inputByteCount);
        inputBuffer.order(ByteOrder.nativeOrder());
        outputBuffer = new float[1][outputSize];

        // Prove that this exact APK runtime can execute the complete model
        // before the app enables scanning. This catches unsupported operators
        // at startup rather than during the user's first photo scan.
        try {
            inputBuffer.clear(); // a new direct buffer is zero-filled
            interpreter.run(inputBuffer, outputBuffer);
            for (int i = 0; i < outputSize; i++) {
                float score = outputBuffer[0][i];
                if (Float.isNaN(score) || Float.isInfinite(score)) {
                    throw new IllegalStateException(
                        "Model startup test returned a non-finite score at " + i
                    );
                }
            }
        } catch (RuntimeException error) {
            interpreter.close();
            throw new IllegalStateException(
                "LiteRT could not execute the model during its startup test: " +
                error.getMessage(), error
            );
        }
    }

    public int getInputByteCount() {
        return inputByteCount;
    }

    public int getOutputSize() {
        return outputSize;
    }

    /**
     * Input is the contiguous 224x224x3 RGB float32 tensor serialized in
     * native byte order by NumPy. Output is one float score per model class.
     */
    public synchronized float[] runBytes(byte[] inputBytes) {
        if (inputBytes == null || inputBytes.length != inputByteCount) {
            throw new IllegalArgumentException(
                "Expected " + inputByteCount + " input bytes, received " +
                (inputBytes == null ? "null" : inputBytes.length)
            );
        }

        inputBuffer.clear();
        inputBuffer.put(inputBytes);
        inputBuffer.rewind();

        interpreter.run(inputBuffer, outputBuffer);
        return outputBuffer[0].clone();
    }

    public synchronized void close() {
        interpreter.close();
    }

    private static String shapeToString(int[] shape) {
        StringBuilder value = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) value.append(",");
            value.append(shape[i]);
        }
        return value.append("]").toString();
    }
}
