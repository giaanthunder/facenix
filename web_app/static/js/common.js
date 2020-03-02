function uuidv4() {
   id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx';

   return id.replace(/[xy]/g, temp);
}

function temp(c) {
   var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
   return v.toString(16);
}

function setCookie(cname, cvalue, exdays) {
   var d = new Date();
   d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
   var expires = "expires="+d.toUTCString();
   document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

function getCookie(cname) {
   var name = cname + "=";
   var ca = document.cookie.split(';');
   for(var i = 0; i < ca.length; i++) {
      var c = ca[i];
      while (c.charAt(0) == ' ') {
         c = c.substring(1);
      }
      if (c.indexOf(name) == 0) {
         return c.substring(name.length, c.length);
      }
   }
   return "";
}

function set_uuid(){
   console.log('an was here')
   var uuid = uuidv4()
   setCookie("uuid", uuid, 30);
}

function del_uuid(){
   uuid = getCookie("uuid");
}

function checkCookie() {
   var user = getCookie("username");
   if (user != "") {
      alert("Welcome again " + user);
   } else {
      user = prompt("Please enter your name:", "");
      if (user != "" && user != null) {
         setCookie("username", user, 365);
      }
   }
}




