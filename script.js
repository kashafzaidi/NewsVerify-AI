
// GET HTML ELEMENTS


const newsInput = document.getElementById("newsInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");

const characterCount = document.getElementById("characterCount");

const loadingCard = document.getElementById("loadingCard");
const resultCard = document.getElementById("resultCard");

const resultTitle = document.getElementById("resultTitle");
const resultIcon = document.getElementById("resultIcon");
const resultStatus = document.getElementById("resultStatus");
const resultDescription = document.getElementById("resultDescription");



// CHARACTER COUNT


newsInput.addEventListener("input", function () {

    const length = newsInput.value.length;

    characterCount.textContent = `${length} characters`;

});



// CLEAR BUTTON


clearBtn.addEventListener("click", function () {

    newsInput.value = "";

    characterCount.textContent = "0 characters";

    resultCard.style.display = "none";

    loadingCard.style.display = "none";

    newsInput.focus();

});



// ANALYZE NEWS


analyzeBtn.addEventListener("click", async function () {

    // Get news entered by user
    const news = newsInput.value.trim();


    
    // EMPTY INPUT CHECK
    

    if (news === "") {

        alert("Please enter some news text first.");

        return;

    }


    
    // SHOW LOADING
    

    loadingCard.style.display = "flex";

    resultCard.style.display = "none";


    try {

        
        // SEND NEWS TO FASTAPI
        

        const response = await fetch(
             "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    news: news
                })
            }
        );


       
        // CHECK SERVER RESPONSE
        

        if (!response.ok) {

            throw new Error(
                `Server returned status ${response.status}`
            );

        }


       
        // GET JSON RESPONSE
       

        const data = await response.json();

        console.log("API Response:", data);


       
        // HIDE LOADING
        

        loadingCard.style.display = "none";


        
        // SHOW RESULT
       

        resultCard.style.display = "block";


        
        // FAKE NEWS
       

        if (data.prediction === "FAKE NEWS") {

            resultTitle.textContent = "Fake News";

            resultIcon.textContent = "✕";

            resultStatus.textContent = "LIKELY FAKE";

            resultDescription.textContent =
                "The ML model classified this content as likely fake based on patterns learned from the training dataset.";

        }


       
        // REAL NEWS
        

        else if (data.prediction === "REAL NEWS") {

            resultTitle.textContent = "Real News";

            resultIcon.textContent = "✓";

            resultStatus.textContent = "LIKELY REAL";

            resultDescription.textContent =
                "The ML model classified this content as likely real based on patterns learned from the training dataset.";

        }


       
        // UNKNOWN RESPONSE
       

        else {

            resultTitle.textContent = "Unknown Result";

            resultIcon.textContent = "?";

            resultStatus.textContent = data.prediction || "UNKNOWN";

            resultDescription.textContent =
                "The model returned an unexpected prediction.";

        }

    }


    
    // CONNECTION ERROR
    
    catch (error) {

        console.error("Prediction Error:", error);

        loadingCard.style.display = "none";

        resultCard.style.display = "none";

        alert(
            "Unable to connect to the ML server. Please make sure FastAPI is running."
        );

    }

});