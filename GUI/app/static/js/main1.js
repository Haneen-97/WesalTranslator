let model;

const modelURL = 'http://localhost:5000/model';
var space;
const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict");
const fileInput = document.getElementById('file');
var inp = document.getElementById('file');

// load the model
const predict = async (modelURL) => {
    if (!model) model = await tf.loadLayersModel(modelURL);
    const files = fileInput.files;

    [...files].map(async (img) => {
        const data = new FormData();
        data.append('file', img);
        // fetch images from prepare request
        const processedImage = await fetch("/api/prepare",
            {
                method: 'POST',
                body: data
            }).then(response => {
                return response.json();
            }).then(result => {
            // convert the image into tensor
                return tf.tensor2d(result['image']);
            });
            // make prediction
        const prediction = model.predict(tf.reshape(processedImage, shape = [1, 64, 64, 1]));
        // get the label of predicated image
        const label = prediction.argMax(axis = 1).dataSync()[0];
        var arabic;
   // map label with arabic letters
    if (label ==0){
    arabic='ع'
    renderImageLabel(img, arabic);
 } else if (label ==1){
    arabic='ال'
    renderImageLabel(img, arabic);
 }else if (label ==2){
    arabic='ا'
    renderImageLabel(img, arabic);
 } else if (label ==3){
    arabic='ب'
    renderImageLabel(img, arabic);
 } else if (label ==4){
    arabic='د'
    renderImageLabel(img, arabic);
 } else if (label ==5){
    arabic='ظ'
    renderImageLabel(img, arabic);
 } else if (label ==6){
    arabic='ض'
    renderImageLabel(img, arabic);
 } else if (label ==7){
    arabic='ف'
    renderImageLabel(img, arabic);
 } else if (label ==8){
    arabic='ق'
    renderImageLabel(img, arabic);
 } else if (label ==9){
    arabic='غ'
    renderImageLabel(img, arabic);
 } else if (label ==10){
    arabic='ه'
    renderImageLabel(img, arabic);
 } else if (label ==11){
    arabic='ح'
    renderImageLabel(img, arabic);
 } else if (label ==12){
    arabic='ج'
    renderImageLabel(img, arabic);
 } else if (label ==13){
    arabic='ك'
    renderImageLabel(img, arabic);
 }else if (label ==14){
    arabic='خ'
    renderImageLabel(img, arabic);
} else if (label ==15){
    arabic='لا'
    renderImageLabel(img, arabic);
 }else if (label ==16){
    arabic='ل'
    renderImageLabel(img, arabic);
} if (label ==17){
    arabic='م'
    renderImageLabel(img, arabic);
 } else if (label ==18){
    arabic='ن'
    renderImageLabel(img, arabic);
 }else if (label ==19){
    arabic='ر'
    renderImageLabel(img, arabic);
 } else if (label ==20){
    arabic='ص'
    renderImageLabel(img, arabic);
 } else if (label ==21){
    arabic='س'
    renderImageLabel(img, arabic);

 } else if (label ==22){
    arabic='ش'
    renderImageLabel(img, arabic);
 } else if (label ==23){
    arabic='ط'
    renderImageLabel(img, arabic);
 } else if (label ==24){
    arabic='ت'
    renderImageLabel(img, arabic);
 } else if (label ==25){
    arabic='ث'
    renderImageLabel(img, arabic);
 } else if (label ==26){
    arabic='ذ'
    renderImageLabel(img, arabic);
 } else if (label ==27){
    arabic='ة'
    renderImageLabel(img, arabic);
 } else if (label ==28){
    arabic='و'
    renderImageLabel(img, arabic);
 } else if (label ==29){
    arabic='ئ'
    renderImageLabel(img, arabic);
 } else if (label ==30){
    arabic='ي'
    renderImageLabel(img, arabic);
 }else if (label == 31){
    arabic='ز'
    renderImageLabel(img, arabic);
}

    })
};
// update the html content
t=" ";
preview.innerHTML +=`${t}`;

const renderImageLabel = (img, label) => {
    const reader = new FileReader();
    var text='';
    reader.onload = () => {
  preview.innerHTML +=`${label}`;
                              };
    reader.readAsDataURL(img);
};



predictButton.addEventListener("click", () => predict(modelURL));
