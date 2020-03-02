function up_btn(){
   set_uuid();
}

function make_att_row(att_name, att_lst, row_num) {
   // Control panel
   var ctrl_panel = document.getElementById("ctrl_panel");

   // Button row
   var btn_row = document.createElement("div");
   btn_row.classList.add("row");
   ctrl_panel.appendChild(btn_row);

   for (var i=0; i<3; i++) {
      label_text = att_name[i]
      // att_id = att_lst[i]
      att_id = "att" + (row_num*3+i).toString()
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

      // att select
      var att_sel = document.createElement("div");
      att_sel.classList.add("col");
      att_sel.classList.add("col-md-6");
      att_sel.style.padding = "1px";
      var temp = document.createElement("select");
      // temp.classList.add("form-control");
      temp.id = att_id;

      opts = ["Unchanged", "Add", "Remove"];
      for (opt of opts) {
         option = document.createElement("option");
         option.innerHTML = opt;
         temp.appendChild(option);
      }
      att_sel.appendChild(temp);
      att_area.appendChild(att_sel);

   }
}

function add_att(){
   var att_name = [
      ["Bald", "Bang", "Black hair"],
      ["Blond hair", "Brown hair", "Eyebrown"],
      ["Eyeglasses", "Male", "Mouth open"],
      ["Mustache", "No Beard", "Pale skin"],
      ["Young", "", ""]
      ];

   var att_lst = [
      ['Bald', 'Bangs', 'Black_Hair'], 
      ['Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows'], 
      ['Eyeglasses', 'Male', 'Mouth_Slightly_Open'], 
      ['Mustache', 'No_Beard', 'Pale_Skin'], 
      ['Young', '', '']
   ]

   for(var i=0; i<5; i++){
      make_att_row(att_name[i], att_lst[i], i);
   }
}

function apply(){
   var value = "";
   for(var i=0; i<13; i++){
      var att = document.getElementById("att"+i.toString());
      var value = value + att.options[att.selectedIndex].text + "_";
   }
   var xhttp = new XMLHttpRequest();
   var cmd = "?cmd=att_mod&value="+value.slice(0,-1)
   console.log(cmd)
   xhttp.open("GET", cmd, true)
   xhttp.send()
   xhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
         var res_img = document.getElementById("res_img");
         res_img.src = this.responseText;
         console.log(this.responseText);
      }
   };
}

function reset(){
   var xhttp = new XMLHttpRequest();
   var cmd = "?cmd=reset"
   console.log(cmd)
   xhttp.open("GET", cmd, true)
   xhttp.send()
   xhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
         var res_img = document.getElementById("res_img");
         res_img.src = this.responseText;
         console.log(this.responseText);

         for(var i=0; i<13; i++){
            var att = document.getElementById("att"+i.toString());
            att.selectedIndex = "0";
         }
      }
   };
}

function download(){
   var res_img = document.getElementById("res_img")
   let a = document.createElement('a');
   a.href = window.location.hostname + res_img.src;
   console.log(a.href)
   a.download = true;
   var event = document.createEvent('Event');
   event.initEvent('click', true, true);
   a.dispatchEvent(event)
}