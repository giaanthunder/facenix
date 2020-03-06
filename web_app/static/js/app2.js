function up_btn(){
   set_uuid();
}

function choose_sample(){
   var xhttp = new XMLHttpRequest();
   var cmd = "?cmd=rand_face";
   console.log(cmd);
   xhttp.open("GET", cmd, true);
   xhttp.send();
   xhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
         var res_img = document.getElementById("res_img");
         res_img.src = this.responseText;
         console.log(this.responseText);
      }
   };   
}

function rand_face(){
   var xhttp = new XMLHttpRequest();
   var cmd = "?cmd=rand_face";
   console.log(cmd);
   xhttp.open("GET", cmd, true);
   xhttp.send();
   xhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
         var res_img = document.getElementById("res_img");
         res_img.src = this.responseText;
         console.log(this.responseText);
      }
   };
}

function make_att_row(att_name, att_lst) {
   // Control panel
   var ctrl_panel = document.getElementById("ctrl_panel");

   // Button row
   var btn_row = document.createElement("div");
   btn_row.classList.add("row");
   ctrl_panel.appendChild(btn_row);

   for (var i=0; i<3; i++) {
      label_text = att_name[i]
      btn_class  = att_lst[i]
      if (label_text == ""){
         break;
      }

      // Single attribute: label | + button | - button
      var att_area_col = document.createElement("div");
      att_area_col.classList.add("col");
      att_area_col.classList.add("col-md-4");
      var att_area = document.createElement("div");
      att_area.classList.add("row");
      att_area_col.appendChild(att_area);
      btn_row.appendChild(att_area_col);

      // label
      var label = document.createElement("div");
      label.classList.add("col");
      label.classList.add("col-md-6");
      label.classList.add("vert_cen");
      var temp = document.createElement("p");
      temp.classList.add("lbl_txt");
      temp.innerText = label_text;
      label.appendChild(temp);
      att_area.appendChild(label);

      // button +
      var add_btn = document.createElement("div");
      add_btn.classList.add("col");
      add_btn.classList.add("col-md-3");
      add_btn.style.padding = "1px";
      var temp = document.createElement("button");
      temp.classList.add("btn");
      temp.classList.add("btn-success");
      temp.classList.add("btn-block");
      temp.classList.add("btn-attr");
      temp.classList.add(btn_class);
      temp.innerText = "+"
      temp.onclick = att_click;
      add_btn.appendChild(temp);
      att_area.appendChild(add_btn);

      // button -
      var rm_btn = document.createElement("div");
      rm_btn.classList.add("col");
      rm_btn.classList.add("col-md-3");
      rm_btn.style.padding = "1px";
      var temp = document.createElement("button");
      temp.classList.add("btn");
      temp.classList.add("btn-success");
      temp.classList.add("btn-block");
      temp.classList.add("btn-attr");
      temp.classList.add(btn_class);
      temp.innerText = "-";
      temp.onclick = att_click;
      rm_btn.appendChild(temp);
      att_area.appendChild(rm_btn);
   }
}

function add_att(){
   console.log("an was here")
   var att_name = [
      'Attractive' , 'Bangs'     , 'Black Hair', 
      'Blond Hair' , 'Brown Hair', 'Eyeglasses', 
      'Goatee'     , 'Gender'      , 'Mouth Open',
      'Narrow Eyes', 'Smiling'   , 'Age',
      'Bald'       , 'Pose'      , ''
   ];

   var att_lst = [
      'Attractive' , 'Bangs'     , 'Black_Hair'         , 
      'Blond_Hair' , 'Brown_Hair', 'Eyeglasses'         , 
      'Goatee'     , 'Male'      , 'Mouth_Slightly_Open',
      'Narrow_Eyes', 'Smiling'   , 'Young',
      'Bald'       , 'Pose'      , ''
   ];

   for(var i=0; i<5; i++){
      make_att_row(att_name.slice(i*3,i*3+3), att_lst.slice(i*3,i*3+3));
   }
}

function att_click(){
   var xhttp = new XMLHttpRequest();
   if (this.innerText == "+") {
      var value = "add";
   } else {
      var value = "minus";
   }

   var cmd = "?cmd=att_mod" + "&att=" + this.classList[4] + "&value=" + value
   console.log(cmd)
   xhttp.open("GET", cmd, true)
   xhttp.send()
   xhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
         var res_img = document.getElementById("res_img");
         res_img.src = this.responseText;
         // console.log(this.responseText);
      }
   };
}

function reset(){
   var xhttp = new XMLHttpRequest();
   var cmd = "?cmd=reset";
   console.log(cmd);
   xhttp.open("GET", cmd, true);
   xhttp.send();
   xhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
         var res_img = document.getElementById("res_img");
         res_img.src = this.responseText;
         console.log(this.responseText);
      }
   };
}

function download(){
   var xhttp = new XMLHttpRequest();
   var cmd = "?cmd=download"
   xhttp.open("GET", cmd, true)
   xhttp.send()
   console.log('request sent')
   xhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
         download_file(this.responseText)
      }
   };
}


