document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");
    const loader = document.getElementById("loader");

    if(form){

        form.addEventListener("submit", function(){

            if(loader){
                loader.style.display = "flex";
            }

        });

    }

});