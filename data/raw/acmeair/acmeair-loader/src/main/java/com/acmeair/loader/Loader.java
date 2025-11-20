package com.acmeair.loader;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.logging.Logger;

import com.acmeair.service.CustomerService;
import com.acmeair.service.FlightService;
import com.acmeair.service.ServiceLocator;

public class Loader {
    private static Logger logger = Logger.getLogger(Loader.class.getName());

    public String queryLoader() {
        String message = System.getProperty("loader.numCustomers");
        if (message == null) {
            logger.info("System property 'loader.numCustomers' not set. Using defaults.");
            lookupDefaults();
            message = System.getProperty("loader.numCustomers");
        }
        return message;
    }

    public String loadDB(long numCustomers) {
        if (numCustomers == -1) {
            return execute();
        } else {
            System.setProperty("loader.numCustomers", Long.toString(numCustomers));
            return execute(numCustomers);
        }
    }

    public static void main(String[] args) {
        // 设置默认加载的用户数量
        System.setProperty("loader.numCustomers", "100");

        Loader loader = new Loader();
        loader.execute();
    }

    private String execute() {
        String numCustomers = System.getProperty("loader.numCustomers");
        if (numCustomers == null) {
            lookupDefaults();
            numCustomers = System.getProperty("loader.numCustomers");
        }
        return execute(Long.parseLong(numCustomers));
    }

    private String execute(long numCustomers) {
        double length = 0;
        try {
            long start = System.currentTimeMillis();
            logger.info("Start loading flights");

            // 创建 Loader 实例
            FlightLoader flightLoader = new FlightLoader();
            CustomerLoader customerLoader = new CustomerLoader();

            // 使用 ServiceLocator 注入默认服务
            FlightService flightService = ServiceLocator.instance().getService(FlightService.class);
            CustomerService customerService = ServiceLocator.instance().getService(CustomerService.class);

            flightLoader.setFlightService(flightService);
            customerLoader.setCustomerService(customerService);

            // 执行加载
            flightLoader.loadFlights();
            logger.info("Start loading " + numCustomers + " customers");
            customerLoader.loadCustomers(numCustomers);

            long stop = System.currentTimeMillis();
            logger.info("Finished loading in " + (stop - start) / 1000.0 + " seconds");
            length = (stop - start) / 1000.0;

        } catch (Exception e) {
            e.printStackTrace();
        }
        return "Loaded flights and " + numCustomers + " customers in " + length + " seconds";
    }

    private void lookupDefaults() {
        Properties props = getProperties();
        String numCustomers = props.getProperty("loader.numCustomers", "100");
        System.setProperty("loader.numCustomers", numCustomers);
    }

    private Properties getProperties() {
        Properties props = new Properties();
        String propFileName = "/loader.properties";
        try {
            InputStream propFileStream = Loader.class.getResourceAsStream(propFileName);
            if (propFileStream != null) {
                props.load(propFileStream);
            }
        } catch (FileNotFoundException e) {
            logger.info("Property file " + propFileName + " not found.");
        } catch (IOException e) {
            logger.info("IOException - Property file " + propFileName + " not found.");
        }
        return props;
    }
}
