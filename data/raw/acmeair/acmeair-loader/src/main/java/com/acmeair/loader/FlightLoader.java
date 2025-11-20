package com.acmeair.loader;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.util.*;
import java.math.*;

import com.acmeair.entities.AirportCodeMapping;
import com.acmeair.service.FlightService;

public class FlightLoader {

    private FlightService flightService; // 默认 null，后面用 setter 注入

    private static final int MAX_FLIGHTS_PER_SEGMENT = 30;

    public FlightLoader() {
        // 默认构造函数
        this.flightService = null;
    }

    // 注入 FlightService
    public void setFlightService(FlightService service) {
        this.flightService = service;
    }

    public void loadFlights() throws Exception {
        if (flightService == null) {
            throw new IllegalStateException("FlightService is not set!");
        }

        InputStream csvInputStream = FlightLoader.class.getResourceAsStream("/mileage.csv");
        LineNumberReader lnr = new LineNumberReader(new InputStreamReader(csvInputStream));
        String line1 = lnr.readLine();
        StringTokenizer st = new StringTokenizer(line1, ",");
        ArrayList<AirportCodeMapping> airports = new ArrayList<>();

        // 读取第一行机场名称
        while (st.hasMoreTokens()) {
            AirportCodeMapping acm = flightService.createAirportCodeMapping(null, st.nextToken());
            airports.add(acm);
        }

        // 读取第二行机场代码
        String line2 = lnr.readLine();
        st = new StringTokenizer(line2, ",");
        int ii = 0;
        while (st.hasMoreTokens()) {
            String airportCode = st.nextToken();
            airports.get(ii).setAirportCode(airportCode);
            ii++;
        }

        // 读取其他航班信息
        String line;
        int flightNumber = 0;
        while ((line = lnr.readLine()) != null && !line.trim().equals("")) {
            st = new StringTokenizer(line, ",");
            String airportName = st.nextToken();
            String airportCode = st.nextToken();
            if (!alreadyInCollection(airportCode, airports)) {
                AirportCodeMapping acm = flightService.createAirportCodeMapping(airportCode, airportName);
                airports.add(acm);
            }
            int indexIntoTopLine = 0;
            while (st.hasMoreTokens()) {
                String milesString = st.nextToken();
                if (milesString.equals("NA")) {
                    indexIntoTopLine++;
                    continue;
                }
                int miles = Integer.parseInt(milesString);
                String toAirport = airports.get(indexIntoTopLine).getAirportCode();
                String flightId = "AA" + flightNumber;
                flightService.storeFlightSegment(flightId, airportCode, toAirport, miles);

                Date now = new Date();
                for (int daysFromNow = 0; daysFromNow < MAX_FLIGHTS_PER_SEGMENT; daysFromNow++) {
                    Calendar c = Calendar.getInstance();
                    c.setTime(now);
                    c.set(Calendar.HOUR_OF_DAY, 0);
                    c.set(Calendar.MINUTE, 0);
                    c.set(Calendar.SECOND, 0);
                    c.set(Calendar.MILLISECOND, 0);
                    c.add(Calendar.DATE, daysFromNow);
                    Date departureTime = c.getTime();
                    Date arrivalTime = getArrivalTime(departureTime, miles);
                    flightService.createNewFlight(flightId, departureTime, arrivalTime,
                            new BigDecimal(500), new BigDecimal(200), 10, 200, "B747");
                }
                flightNumber++;
                indexIntoTopLine++;
            }
        }

        // 保存机场映射
        for (AirportCodeMapping acm : airports) {
            flightService.storeAirportMapping(acm);
        }

        lnr.close();
    }

    private static Date getArrivalTime(Date departureTime, int mileage) {
        double averageSpeed = 600.0; // 600 miles/hours
        double hours = (double) mileage / averageSpeed;
        double partsOfHour = hours % 1.0;
        int minutes = (int) (60.0 * partsOfHour);
        Calendar c = Calendar.getInstance();
        c.setTime(departureTime);
        c.add(Calendar.HOUR, (int) hours);
        c.add(Calendar.MINUTE, minutes);
        return c.getTime();
    }

    private static boolean alreadyInCollection(String airportCode, ArrayList<AirportCodeMapping> airports) {
        for (AirportCodeMapping acm : airports) {
            if (acm.getAirportCode().equals(airportCode)) {
                return true;
            }
        }
        return false;
    }
}
