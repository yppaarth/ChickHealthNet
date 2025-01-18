from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path



class Training:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None
        self.steps_per_epoch = None
        self.validation_steps = None

    def get_base_model(self):
        # Load the base model and compile it
        print(f"Loading base model from {self.config.updated_base_model_path}")
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print("Base model loaded and compiled successfully.")

        # Perform dummy training to initialize metrics
        dummy_input = tf.random.uniform((1, *self.config.params_image_size))
        dummy_output = tf.constant([[1, 0]])  # Replace with your output format
        self.model.train_on_batch(dummy_input, dummy_output)
        print(f"Metrics initialized: {self.model.metrics_names}")

    def train_valid_generator(self):
        # Data generator configuration
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation generator
        print("Creating validation data generator...")
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Training generator with or without augmentation
        print("Creating training data generator...")
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        print("Data generators created successfully.")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        print(f"Saving trained model to {path}")
        model.save(path)
        print("Model saved successfully.")

    def train(self, callback_list: list):
        # Ensure model and data generators are ready
        assert self.model is not None, "Model has not been initialized."
        assert self.train_generator is not None, "Training data generator is not set up."
        assert self.valid_generator is not None, "Validation data generator is not set up."

        # Calculate steps per epoch
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        print(f"Starting training for {self.config.params_epochs} epochs...")
        print(f"Steps per epoch: {self.steps_per_epoch}, Validation steps: {self.validation_steps}")

        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        print("Training completed.")
        return history