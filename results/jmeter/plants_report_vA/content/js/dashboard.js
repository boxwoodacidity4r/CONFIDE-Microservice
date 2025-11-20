/*
   Licensed to the Apache Software Foundation (ASF) under one or more
   contributor license agreements.  See the NOTICE file distributed with
   this work for additional information regarding copyright ownership.
   The ASF licenses this file to You under the Apache License, Version 2.0
   (the "License"); you may not use this file except in compliance with
   the License.  You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
var showControllersOnly = false;
var seriesFilter = "";
var filtersOnlySampleSeries = true;

/*
 * Add header in statistics table to group metrics by category
 * format
 *
 */
function summaryTableHeader(header) {
    var newRow = header.insertRow(-1);
    newRow.className = "tablesorter-no-sort";
    var cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 1;
    cell.innerHTML = "Requests";
    newRow.appendChild(cell);

    cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 3;
    cell.innerHTML = "Executions";
    newRow.appendChild(cell);

    cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 7;
    cell.innerHTML = "Response Times (ms)";
    newRow.appendChild(cell);

    cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 1;
    cell.innerHTML = "Throughput";
    newRow.appendChild(cell);

    cell = document.createElement('th');
    cell.setAttribute("data-sorter", false);
    cell.colSpan = 2;
    cell.innerHTML = "Network (KB/sec)";
    newRow.appendChild(cell);
}

/*
 * Populates the table identified by id parameter with the specified data and
 * format
 *
 */
function createTable(table, info, formatter, defaultSorts, seriesIndex, headerCreator) {
    var tableRef = table[0];

    // Create header and populate it with data.titles array
    var header = tableRef.createTHead();

    // Call callback is available
    if(headerCreator) {
        headerCreator(header);
    }

    var newRow = header.insertRow(-1);
    for (var index = 0; index < info.titles.length; index++) {
        var cell = document.createElement('th');
        cell.innerHTML = info.titles[index];
        newRow.appendChild(cell);
    }

    var tBody;

    // Create overall body if defined
    if(info.overall){
        tBody = document.createElement('tbody');
        tBody.className = "tablesorter-no-sort";
        tableRef.appendChild(tBody);
        var newRow = tBody.insertRow(-1);
        var data = info.overall.data;
        for(var index=0;index < data.length; index++){
            var cell = newRow.insertCell(-1);
            cell.innerHTML = formatter ? formatter(index, data[index]): data[index];
        }
    }

    // Create regular body
    tBody = document.createElement('tbody');
    tableRef.appendChild(tBody);

    var regexp;
    if(seriesFilter) {
        regexp = new RegExp(seriesFilter, 'i');
    }
    // Populate body with data.items array
    for(var index=0; index < info.items.length; index++){
        var item = info.items[index];
        if((!regexp || filtersOnlySampleSeries && !info.supportsControllersDiscrimination || regexp.test(item.data[seriesIndex]))
                &&
                (!showControllersOnly || !info.supportsControllersDiscrimination || item.isController)){
            if(item.data.length > 0) {
                var newRow = tBody.insertRow(-1);
                for(var col=0; col < item.data.length; col++){
                    var cell = newRow.insertCell(-1);
                    cell.innerHTML = formatter ? formatter(col, item.data[col]) : item.data[col];
                }
            }
        }
    }

    // Add support of columns sort
    table.tablesorter({sortList : defaultSorts});
}

$(document).ready(function() {

    // Customize table sorter default options
    $.extend( $.tablesorter.defaults, {
        theme: 'blue',
        cssInfoBlock: "tablesorter-no-sort",
        widthFixed: true,
        widgets: ['zebra']
    });

    var data = {"OkPercent": 100.0, "KoPercent": 0.0};
    var dataset = [
        {
            "label" : "FAIL",
            "data" : data.KoPercent,
            "color" : "#FF6347"
        },
        {
            "label" : "PASS",
            "data" : data.OkPercent,
            "color" : "#9ACD32"
        }];
    $.plot($("#flot-requests-summary"), dataset, {
        series : {
            pie : {
                show : true,
                radius : 1,
                label : {
                    show : true,
                    radius : 3 / 4,
                    formatter : function(label, series) {
                        return '<div style="font-size:8pt;text-align:center;padding:2px;color:white;">'
                            + label
                            + '<br/>'
                            + Math.round10(series.percent, -2)
                            + '%</div>';
                    },
                    background : {
                        opacity : 0.5,
                        color : '#000'
                    }
                }
            }
        },
        legend : {
            show : true
        }
    });

    // Creates APDEX table
    createTable($("#apdexTable"), {"supportsControllersDiscrimination": true, "overall": {"data": [0.9846153846153847, 500, 1500, "Total"], "isController": false}, "titles": ["Apdex", "T (Toleration threshold)", "F (Frustration threshold)", "Label"], "items": [{"data": [1.0, 500, 1500, "Product Page (POST - view product / add form)"], "isController": false}, {"data": [1.0, 500, 1500, "Help Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "Order Info (POST)"], "isController": false}, {"data": [1.0, 500, 1500, "Admin Home Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "Final Checkout (POST)"], "isController": false}, {"data": [1.0, 500, 1500, "Add To Cart (POST)"], "isController": false}, {"data": [0.75, 500, 1500, "Backorder Admin Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "Admin Actions Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "Shopping Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "Cart Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "User Login (POST)"], "isController": false}, {"data": [0.9, 500, 1500, "Home / Promo Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "Order Done Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "Supplier Config Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "Account Page (GET)"], "isController": false}, {"data": [1.0, 500, 1500, "Admin Banner Page (GET)"], "isController": false}]}, function(index, item){
        switch(index){
            case 0:
                item = item.toFixed(3);
                break;
            case 1:
            case 2:
                item = formatDuration(item);
                break;
        }
        return item;
    }, [[0, 0]], 3);

    // Create statistics table
    createTable($("#statisticsTable"), {"supportsControllersDiscrimination": true, "overall": {"data": ["Total", 65, 0, 0.0, 91.55384615384617, 7, 836, 61.0, 139.2, 208.0999999999999, 836.0, 5.772133913506793, 98.10260176161087, 4.539398282790161], "isController": false}, "titles": ["Label", "#Samples", "FAIL", "Error %", "Average", "Min", "Max", "Median", "90th pct", "95th pct", "99th pct", "Transactions/s", "Received", "Sent"], "items": [{"data": ["Product Page (POST - view product / add form)", 5, 0, 0.0, 68.4, 61, 80, 68.0, 80.0, 80.0, 80.0, 1.658374792703151, 18.58545812603648, 1.462414490049751], "isController": false}, {"data": ["Help Page (GET)", 5, 0, 0.0, 50.4, 42, 60, 50.0, 60.0, 60.0, 60.0, 1.3065064018813692, 13.516727364776587, 1.0704676476352235], "isController": false}, {"data": ["Order Info (POST)", 5, 0, 0.0, 114.0, 79, 172, 103.0, 172.0, 172.0, 172.0, 1.7655367231638417, 166.03217690677968, 1.560362045374294], "isController": false}, {"data": ["Admin Home Page (GET)", 2, 0, 0.0, 111.0, 7, 215, 111.0, 215.0, 215.0, 215.0, 1.0970927043335161, 2.631308283049918, 0.4114097641250686], "isController": false}, {"data": ["Final Checkout (POST)", 5, 0, 0.0, 78.4, 63, 97, 78.0, 97.0, 97.0, 97.0, 1.4806040864672787, 23.33686519099793, 1.3157712096535388], "isController": false}, {"data": ["Add To Cart (POST)", 5, 0, 0.0, 52.2, 49, 55, 52.0, 55.0, 55.0, 55.0, 1.8096272167933405, 17.131373280854145, 1.597561527325371], "isController": false}, {"data": ["Backorder Admin Page (GET)", 2, 0, 0.0, 423.5, 11, 836, 423.5, 836.0, 836.0, 836.0, 1.0454783063251436, 11.864749411918453, 0.7483746079456352], "isController": false}, {"data": ["Admin Actions Page (GET)", 2, 0, 0.0, 35.0, 11, 59, 35.0, 59.0, 59.0, 59.0, 1.1068068622025455, 3.2609729524073052, 0.791193967902601], "isController": false}, {"data": ["Shopping Page (GET)", 5, 0, 0.0, 62.8, 47, 86, 60.0, 86.0, 86.0, 86.0, 1.7024174327545114, 16.11644003234593, 1.401501851378958], "isController": false}, {"data": ["Cart Page (GET)", 5, 0, 0.0, 80.2, 60, 132, 70.0, 132.0, 132.0, 132.0, 1.7599436818021823, 19.099857554558252, 1.4419851064765927], "isController": false}, {"data": ["User Login (POST)", 5, 0, 0.0, 80.6, 45, 192, 55.0, 192.0, 192.0, 192.0, 2.046663937781416, 22.591252430413427, 1.8008244218174376], "isController": false}, {"data": ["Home / Promo Page (GET)", 5, 0, 0.0, 228.4, 60, 783, 99.0, 783.0, 783.0, 783.0, 1.3751375137513753, 18.537498281078108, 0.5143336599284928], "isController": false}, {"data": ["Order Done Page (GET)", 5, 0, 0.0, 44.4, 39, 53, 41.0, 53.0, 53.0, 53.0, 1.461133839859731, 13.847953042811222, 1.2042939070718877], "isController": false}, {"data": ["Supplier Config Page (GET)", 2, 0, 0.0, 80.0, 10, 150, 80.0, 150.0, 150.0, 150.0, 2.642007926023778, 20.955457397622194, 2.0898695508586527], "isController": false}, {"data": ["Account Page (GET)", 5, 0, 0.0, 64.6, 55, 78, 63.0, 78.0, 78.0, 78.0, 1.2903225806451613, 19.110383064516128, 1.0609879032258065], "isController": false}, {"data": ["Admin Banner Page (GET)", 2, 0, 0.0, 15.0, 9, 21, 15.0, 21.0, 21.0, 21.0, 1.1983223487118035, 3.4416660425404433, 0.8554430047932894], "isController": false}]}, function(index, item){
        switch(index){
            // Errors pct
            case 3:
                item = item.toFixed(2) + '%';
                break;
            // Mean
            case 4:
            // Mean
            case 7:
            // Median
            case 8:
            // Percentile 1
            case 9:
            // Percentile 2
            case 10:
            // Percentile 3
            case 11:
            // Throughput
            case 12:
            // Kbytes/s
            case 13:
            // Sent Kbytes/s
                item = item.toFixed(2);
                break;
        }
        return item;
    }, [[0, 0]], 0, summaryTableHeader);

    // Create error table
    createTable($("#errorsTable"), {"supportsControllersDiscrimination": false, "titles": ["Type of error", "Number of errors", "% in errors", "% in all samples"], "items": []}, function(index, item){
        switch(index){
            case 2:
            case 3:
                item = item.toFixed(2) + '%';
                break;
        }
        return item;
    }, [[1, 1]]);

        // Create top5 errors by sampler
    createTable($("#top5ErrorsBySamplerTable"), {"supportsControllersDiscrimination": false, "overall": {"data": ["Total", 65, 0, "", "", "", "", "", "", "", "", "", ""], "isController": false}, "titles": ["Sample", "#Samples", "#Errors", "Error", "#Errors", "Error", "#Errors", "Error", "#Errors", "Error", "#Errors", "Error", "#Errors"], "items": [{"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}, {"data": [], "isController": false}]}, function(index, item){
        return item;
    }, [[0, 0]], 0);

});
