package com.example.fruitsclassifierapp;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;

import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.Manifest;

import com.example.fruitsclassifierapp.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;


public class MainActivity extends AppCompatActivity {

    TextView result, confidence2;
    ImageView imageView;
    Button picture;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        result = findViewById(R.id.result);
        confidence2 = findViewById(R.id.confidence);
        imageView = findViewById(R.id.image1);
        picture = findViewById(R.id.button);


        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }


    public void ClassifyImage(Bitmap image) {
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*imageSize*imageSize*3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int [] intvalues = new int[imageSize*imageSize];
            image.getPixels(intvalues,0, image.getWidth(),0,0,image.getWidth(), image.getHeight());
            int pixel = 0;
            for(int i=0; i<imageSize; i++){
                for(int j=0; j<imageSize; j++){
                    int val = intvalues[pixel++];
                    byteBuffer.putFloat(((val>>16) & 0xFF)*(1.f / 255.f));
                    byteBuffer.putFloat(((val>>8) & 0xFF)*(1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 225.f));
                }
            }
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float [] confidence = outputFeature0.getFloatArray();
            int maxPos=0;
            float maxConfidence =0;
            for(int i=0;i<confidence.length;i++){
                if(confidence[i]>maxConfidence){
                    maxConfidence = confidence[i];
                    maxPos=i;
                }
            }

            String[] classes = {"Banana","orange","Apple", "Grapes"};
            result.setText(classes[maxPos]);

            String s = "";
            for(int i = 0; i < classes.length; i++) {
                s += String.format("%s: %.2f%%\n", classes[i], confidence[i] * 100);
            }

            confidence2.setText(s);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /*
    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int Dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, Dimension, Dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image,imageSize,imageSize, false);
            ClassifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    } */
    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK) {
            if (data != null && data.getExtras() != null) {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                if (image != null) {
                    int Dimension = Math.min(image.getWidth(), image.getHeight());
                    image = ThumbnailUtils.extractThumbnail(image, Dimension, Dimension);
                    imageView.setImageBitmap(image);

                    image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                    ClassifyImage(image);
                } else {
                    // Handle the case where image is null
                    Log.e("MainActivity", "No image data received");
                }
            } else {
                // Handle the case where extras are null
                Log.e("MainActivity", "No extras found in the Intent");
            }
        }
    }


}
