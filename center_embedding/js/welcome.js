var welcome = {};

// --------------  things that vary from task to task --------------

welcome.task = {};
welcome.task.blurb = 'This is a short psychological study investigating how people learn languages.';
welcome.task.time = '15 minutes';
welcome.task.pay = 'US$4.00';


// --------------  things that vary between ethics approvals --------------

welcome.ethics = {};
welcome.ethics.approval = '5932';
welcome.ethics.name = 'Sources of Difficulty in Language Processing and Learning';
welcome.ethics.selection = 'You are invited to participate in a study of how humans learn languages.  We hope to learn what factors  make languages easy or difficult to learn. You were selected as a possible participant in this study because you accepted our HIT on Amazon Mechanical Turk.';
welcome.ethics.description = 'If you decide to participate, we will present you with some examples from an alien language, which you will have to repeat back. We will then test you on what you have learned from these examples. The task should take approximately ' + welcome.task.time + ' to complete including reading time.';

// ----------------------- function to start the task ------------------
welcome.run = function() {
    document.getElementById("welcome").innerHTML =
        welcome.section.header +
        welcome.section.start +
        welcome.section.consent
}

// ------------- actions to take at the end of each click ----------------
welcome.click = {};
welcome.click.start = function() {
    welcome.helpers.setDisplay('start', 'none');
    welcome.helpers.setDisplay('consent', '');
    welcome.helpers.setHeader(' ');   
}
welcome.click.consent = function() {
    welcome.helpers.setDisplay('consent', 'none');
    welcome.helpers.setDisplay('header', 'none');
    startExperiment(); // start the jsPsych experiment
}


// ------------- html for the various sections ----------------
welcome.section = {};
welcome.section.header =
    '<!-- ####################### Heading ####################### -->' +
    '<a name="top"></a>' +
    '<h1 style="text-align:center; width:1200px" id="header" class="header">' +
    '   &nbsp; JHU Cognitive Science</h1>';


welcome.section.start =
    '<!-- ####################### Start page ####################### -->' +
    '<div class="start" style="width: 1000px">' +
    '<div class="start" style="text-align:left; border:0px solid; padding:10px;' +
    '                          width:800px; float:right; font-size:90%">' +
	'<p>This is an online demo for the experiment in the paper "Infinite use of finite means? Evaluating the generalization of center embedding learned from an artificial grammar" by R. Thomas McCoy, Jennifer Culbertson, Paul Smolensky, and G&eacute;raldine Legendre.</p>' +
	'<p>Parts of the code for this experiment come from the following template by Danielle Navarro: https://github.com/djnavarro/blankex</p>' +
	'<p>Aside from some aspects of the instructions that have been modified for this public release, the experiment appears exactly as it did to participants.</p>' +
	'<br><div stle="text-align:center;"><p style="text-align:center">****************</p></div><br>' +
    '<p>Thanks for accepting the HIT. ' + welcome.task.blurb + ' It involves the following steps:</p>' +
    '<ol>' +
    '<li> Because this is a University research project, we ask for your informed consent. ' +
    '     (The format of the consent form is a  standard university document, so it sometimes ' +
    '     looks a little weird on MTurk)<br></li>' +
    '<li> The study then explains how to do the task.<br></li>' +
    '<li> Next comes the experiment itself.<br></li>' +
    '<li> At the end, we will give you the completion code you need to get paid for the HIT.</li>' +
    '</ol>' +
    '<p>The total time taken should be about ' + welcome.task.time + '. Please <u>do not</u> use the "back" ' +
    '   button on your browser or close the window until you reach the end and receive your completion ' +
    '   code. This is very likely to break the experiment and may make it difficult for you to get paid.' +
    '   However, if something does go wrong, please contact us! When you are ready to begin, click on' +
    '   the "start" button below.</p>' +
    '<!-- Next button for the splash page -->' +
    '<p align="center"> <input type="button" id="splashButton" class="start jspsych-btn" ' +
    'value="Start!" onclick="welcome.click.start()"> </p>' +
    '</div>' +
    '</div>';

welcome.section.consent =
    '   <!-- ####################### Consent ####################### -->' +
    '   <div class="consent" style="display:none; width:1000px">' +
    '       <!-- Text box for the splash page -->' +
    '       <div class="consent" style="text-align:left; border:0px solid; padding:10px;  width:800px; font-size:90%; float:right">' +
    '           <p align="center"><b>JOHNS HOPKINS UNIVERSITY</b><br>' +
    '           <p><b>INFORMED CONSENT FORM:</b></p>' +
    '           <p>In the actual experiment as presented to participants, our informed consent form appeared here. For this public online release of the experiment, we have removed the text of this form to avoid releasing our PI\'s personal contact information, which is included in the consent form.' +
    '           <p align="center">' +
    '           <input type="button" id="consentButton" class="consent jspsych-btn" value="Continue" onclick="welcome.click.consent()" >' +
    '           </p>' +
    '       </div><br><br></div>';




// ----------------------- helper functions ------------------

welcome.helpers = {};
welcome.helpers.getRadioButton = function(name) { // get the value of a radio button
    var i, radios = document.getElementsByName(name);
    for (i = 0; i < radios.length; i = i + 1) {
        if (radios[i].checked) {
            return (radios[i].value);
        }
    }
    return ("NA");
}
welcome.helpers.setDisplay = function(theClass, theValue) { // toggle display status
    var i, classElements = document.getElementsByClassName(theClass);
    for (i = 0; i < classElements.length; i = i + 1) {
        classElements[i].style.display = theValue;
    }
}
welcome.helpers.setVisibility = function(theClass, theValue) { // toggle visibility
    var i, classElements = document.getElementsByClassName(theClass);
    for (i = 0; i < classElements.length; i = i + 1) {
        classElements[i].style.visibility = theValue;
    }
}
welcome.helpers.setHeader = function(theValue) { // alter the header
    document.getElementById("header").innerText = theValue;
}




