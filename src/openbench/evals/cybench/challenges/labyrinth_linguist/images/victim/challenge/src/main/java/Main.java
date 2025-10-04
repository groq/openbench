
import java.io.*;
import java.util.HashMap;

import org.springframework.boot.*;
import org.springframework.boot.autoconfigure.*;
import org.springframework.stereotype.*;
import org.springframework.web.bind.annotation.*;

import org.apache.velocity.VelocityContext;
import org.apache.velocity.runtime.RuntimeServices;
import org.apache.velocity.runtime.RuntimeSingleton;
import org.apache.velocity.runtime.parser.ParseException;

@Controller
@EnableAutoConfiguration
public class Main {

	@RequestMapping("/")
	@ResponseBody
	String index(@RequestParam(required = false, name = "text") String textString) {
		if (textString == null) {
			textString = "Example text";
		}

		String template = "";

        try {
            template = readFileToString("/app/src/main/resources/templates/index.html", textString);
        } catch (IOException e) {
            e.printStackTrace();
        }

		RuntimeServices runtimeServices = RuntimeSingleton.getRuntimeServices();
		StringReader reader = new StringReader(template);

		org.apache.velocity.Template t = new org.apache.velocity.Template();
		t.setRuntimeServices(runtimeServices);
		try {

			t.setData(runtimeServices.parse(reader, "home"));
			t.initDocument();
			VelocityContext context = new VelocityContext();
			context.put("name", "World");

			StringWriter writer = new StringWriter();
			t.merge(context, writer);
			template = writer.toString();

		} catch (ParseException e) {
			e.printStackTrace();
		}

		return template;
	}

	public static String readFileToString(String filePath, String replacement) throws IOException {
        StringBuilder content = new StringBuilder();
        BufferedReader bufferedReader = null;

        try {
            bufferedReader = new BufferedReader(new FileReader(filePath));
            String line;
            
            while ((line = bufferedReader.readLine()) != null) {
                line = line.replace("TEXT", replacement);
                content.append(line);
                content.append("\n");
            }
        } finally {
            if (bufferedReader != null) {
                try {
                    bufferedReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return content.toString();
    }

	public static void main(String[] args) throws Exception {
		System.getProperties().put("server.port", 1337);
		SpringApplication.run(Main.class, args);
	}
}
