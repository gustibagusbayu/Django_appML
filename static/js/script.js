$(function() {
    $(document).on("click", "#deleterows", function(){
        var deleterowsfromtable=$(".selectedTable input:checked").parents("tr").remove();
    })
})





