#!/bin/sh
################################################################################
# DO NOT ALTER THIS LINE
VERSION=54
################################################################################

SERVER="https://floods.ssec.wisc.edu"
KEY=
URI=

IS_HELP=0
if [ -n "$1" -a "$1" = "-h" ]; then
        IS_HELP=1
fi
CHECK_SH=`basename ${SHELL}`
if [ ${IS_HELP} -eq 0 -a \
     "${CHECK_SH}" != "sh" -a \
     "${CHECK_SH}" != "ash" -a \
     "${CHECK_SH}" != "bash" ]; then
	echo "Error: Bad interpreter: \"${CHECK_SH}\""
	echo "Please modify the first line of this script to use sh, ash, or bash"
	echo "Or call explicity with:"
	echo "	/bin/sh $0 $@"
	exit 6
fi

CURL=`which curl 2>/dev/null`
if [ -z "${CURL}" ]; then
	echo "Error: Cannot find \"curl\""
	exit 5
fi
CHECK_CURL=`${CURL} --version |head -1 |awk '{print $2}' |awk -F. '{print $1}'`
if [ -z "${CHECK_CURL}" -o ${CHECK_CURL} -lt 7 ]; then
	echo "Error: \"curl\" version must be 7+"
	exit 5
fi

BASE64=`which base64 2>/dev/null`
TIMEOUT_BIN=`which timeout 2>/dev/null`
TIMEOUT=
if [ -n "${TIMEOUT_BIN}" ]; then
	TIMEOUT="${TIMEOUT_BIN} -s 9 600"
fi
TMPDIR=/tmp/re_upload.$$

showHelp() {
	SELF=`basename $0`
	echo ""
	echo "$0 [-hupjtcg1dxv] [-f [hours]] [-p [part]] [-k key|file] [-l \"URI\"] [-s server:port] file [name] [date] [time]"
	echo ""
	echo "   -h: Show help"
	echo "   -u: Check for update"
	echo "       Version: ${VERSION}"
	echo ""
	echo "   -p: Designate file as part of a larger product (part # optional)"
	echo "   -f: Force overwrite if there is a conflict (hour range optional)"
	echo "   -j: Use JPEG compression (requires GDAL tools)"
	echo "   -t: Do not timeout"
	echo ""
	echo "   -k: Specify the upload key or file"
	echo "   -l: Specify a URI for download of the original data"
	echo ""
	echo "   -c: Add random sleep to mitigate concurrent uploads (eg. cron jobs)"
	echo "   -g: Send through public gateway"
	echo "   -s: Specify the server and port"
	echo "       Default: ${SERVER}"
	echo ""
	echo "   -1: Do not retry on failure"
	echo "   -d: Delete file on successful upload"
	echo "   -x: Print target server and port (do not upload)"
	echo "   -v: Be verbose"
	echo ""
	echo " file: Path to file"
	echo "       Format: /path/to/[name]_[YYYYMMDD]_[HHMMSS].???"
	echo ""
	echo " name: Specify the product name"
	echo "       Required when the file name does not contain [name]"
	echo "       Format: Cannot contain '_'"
	echo " date: Specify the date"
	echo "       Required when the file name does not contain [date]"
	echo "       Format: YYYYMMDD"
	echo " time: Specify the time"
	echo "       Required when the file name does not contain [time]"
	echo "       Format: HHMMSS"
	echo ""
}

checkUpdate() {
	DL_URL="${SERVER}/upload/re_upload"
	URL="$DL_URL?version=${VERSION}"
	VERSION_CHECK=`${CURL} -L --insecure -s ${URL}`
	if [ -n "${VERSION_CHECK}" ]; then
		echo "A new version is available at:"
		echo "    ${DL_URL}"
		echo "Download with:"
		echo "    ${CURL} -L ${DL_URL} -o $0"
		if [ -n "${KEY}" ]; then
			echo "WARNING: Custom upload key must be set manually:"
			echo "    KEY=${KEY}"
		fi
		if [ -n "${URI}" ]; then
			echo "WARNING: Custom upload URI must be set manually:"
			echo "    URI=${URI}"
		fi
	else
		echo "This version is up to date"
	fi
}

# Cleanup
cleanup() {
	rm -Rf "${TMPDIR}"
	if [ ${RE_DELETE} -ne 0 ]; then
		if [ ${RE_VERBOSE} -ne 0 ]; then
			echo "Deleting ${RE_FILE_NAME}"
		fi
		rm -f ${RE_FILE}
	fi
}
trap cleanup SIGHUP SIGINT SIGTERM

# Get the options from the command line
RE_ONCE=0
RE_FORCE=0
RE_PART=0
RE_SLEEP=0
RE_PUBLIC=0
RE_VERBOSE=0
RE_PRINT=0
RE_JPEG=0
RE_DELETE=0
if [ -z "$1" ]; then
	showHelp
	exit 1
fi
while getopts 1hufpjtcrgdxvk:l:s: o; do
	case "$o" in
	1)	RE_ONCE=1
		;;
	h)	showHelp
		exit 1
		;;
	u)	checkUpdate
		exit 1
		;;
        f)      eval NEXTIND=\${$OPTIND}
                if [[ -n "${NEXTIND}" && ${NEXTIND} =~ ^[0-9]+$ && ! -f ${NEXTIND} ]]; then
                        OPTIND=$((OPTIND + 1))
                        RE_FORCE=${NEXTIND}
                else
                        RE_FORCE=-1
                fi
                ;;
	p)	eval NEXTIND=\${$OPTIND}
		if [[ -n "${NEXTIND}" && ${NEXTIND} =~ ^[0-9]+$ && ! -f ${NEXTIND} ]]; then
			OPTIND=$((OPTIND + 1))
			RE_PART=${NEXTIND}
		else
			RE_PART=-1
		fi
		;;
	c)	RE_SLEEP=1
		;;
	r)	RE_PUBLIC=1
		;;
	g)	RE_PUBLIC=1
		;;
	v)	RE_VERBOSE=1
		;;
	x)	RE_PRINT=1
		;;
	j)	RE_JPEG=1
		;;
	d)	RE_DELETE=1
		;;
	t)	TIMEOUT=
		;;
	k)	KEY=${OPTARG}
		;;
	l)	URI=${OPTARG}
		if [ -z "${BASE64}" ]; then
			echo "Error: Cannot find \"base64\""
			exit 5
		fi
		;;
	s)	SERVER=${OPTARG}
		;;
	*)	showHelp
		exit 1
		;;
	esac
done
shift $((${OPTIND} -1))

# Insert http:// if not part of SERVER
if [[ ! ${SERVER} =~ ^http ]]; then
	SERVER="http://${SERVER}"
fi
SERVER_GW="${SERVER}"

# Set our variables
RE_FILE=$1
RE_NAME=$2
RE_DATE=$3
RE_TIME=$4

# Does the file exist?
if [ ! -f "${RE_FILE}" -a ${RE_PRINT} -eq 0 ]; then
	echo "ERROR: Could not find file: ${RE_FILE}"
	exit 4
fi

# Is the key a file?
if [ -n "${KEY}" -a -f "${KEY}" ]; then
	KEY=`cat "${KEY}"`
fi

# Set the defaults for sending
RE_FILE_DIR=`dirname "${RE_FILE}"`
RE_FILE_NAME=`basename "${RE_FILE}"`
RE_FILE_PARTS=`echo "${RE_FILE_NAME}" |awk -F. '{print $1}'`

# Verify the product name
CHECK_NAME=${RE_NAME}
if [ -z "${CHECK_NAME}" ]; then
	CHECK_NAME=`echo ${RE_FILE_PARTS} |awk -F_ '{print $1}'`
fi
if [ -n "${CHECK_NAME}" ]; then
	match=`expr "${CHECK_NAME}" : '\([a-zA-Z0-9\-]\{1,\}\)'`
	if [ "$match" != "${CHECK_NAME}" ]; then
		echo ""
		echo "ERROR: Invalid product name"
		showHelp
		exit 4
	fi
fi

# Verify the product date
CHECK_DATE=${RE_DATE}
if [ -z "${CHECK_DATE}" ]; then
	CHECK_DATE=`echo ${RE_FILE_PARTS} |awk -F_ '{print $2}'`
fi
if [ -n "${CHECK_DATE}" ]; then
	match=`expr "${CHECK_DATE}" : '\([0-9]\{7,8\}\)'`
	if [ "$match" != "${CHECK_DATE}" ]; then
		echo ""
		echo "ERROR: Invalid product date"
		showHelp
		exit 4
	fi
fi

# Verify the product time
CHECK_TIME=${RE_TIME}
if [ -z "${CHECK_TIME}" ]; then
	CHECK_TIME=`echo ${RE_FILE_PARTS} |awk -F_ '{print $3}'`
fi
if [ -n "${CHECK_TIME}" ]; then
	match=`expr "${CHECK_TIME}" : '\([0-9]\{6\}\)'`
	if [ "$match" != "${CHECK_TIME}" ]; then
		echo ""
		echo "ERROR: Invalid product time"
		showHelp
		exit 4
	fi
fi

# Get the direct upload name (unless -g was specified)
if [ ${RE_PUBLIC} -eq 0 ]; then
	SERVER_DIRECT=
	if [ -n "${RE_NAME}" ]; then
		SERVER_DIRECT=`${CURL} -L --insecure -s ${SERVER}/upload/re_upload?name=${RE_NAME}`
	else
		SERVER_DIRECT=`${CURL} -L --insecure -s ${SERVER}/upload/re_upload?file=${RE_FILE_NAME}`
	fi
	if [ -n "${SERVER_DIRECT}" ]; then
		SERVER=${SERVER_DIRECT}
		if [[ ! ${SERVER} =~ http* ]]; then
			SERVER="http://${SERVER}"
		fi
	else
		if [ ${RE_VERBOSE} -ne 0 ]; then
			echo "WARNING: Could not determine the direct URL for proxy upload"
		fi
	fi

	# Test the direct upload
	if [ ${RE_VERBOSE} -ne 0 ]; then
		echo "Testing direct connection to ${SERVER}..."
	fi
	SERVER_TEST=`${CURL} -L --max-time 5 --insecure -s ${SERVER}/api/version`
	if [[ -z "${SERVER_TEST}" || ! ${SERVER_TEST} =~ ^[\.0-9]+$ ]]; then
		if [ ${RE_VERBOSE} -ne 0 ]; then
			echo "WARNING: Could not connect directly, using gateway"
		fi
		SERVER="${SERVER_GW}"
	fi
fi

# Print
if [ ${RE_PRINT} -ne 0 ]; then
	echo ""
	echo "   File: ${RE_FILE}"
	echo ""
	echo "Product: ${CHECK_NAME}"
	echo "   Date: ${CHECK_DATE}"
	echo "   Time: ${CHECK_TIME}"
	echo ""
	echo " Target: ${SERVER}"
	echo ""
	exit 0
fi

# Sleep up to 15 seconds if asked to
if [ ${RE_SLEEP} -ne 0 ]; then
	SLEEP=$((${RANDOM} * 15 / 32767))
	echo "Sleeping for ${SLEEP} seconds..."
	sleep ${SLEEP}
fi

# See if we can do compression
if [ ${RE_JPEG} -ne 0 ]; then
	GDAL_TRANSLATE=`which gdal_translate 2>/dev/null`
	if [ -z "${GDAL_TRANSLATE}" ]; then
		echo "Warning: Cannot find \"gdal_translate\", JPEG compression disabled"
		RE_JPEG=0
	fi
	if [[ ! ${RE_FILE_NAME} =~ .tif$ ]]; then
		echo "Warning: JPEG compression only applies to GeoTIFFs"
		RE_JPEG=0
	fi
fi

# Did we ask for compression?
if [ ${RE_JPEG} -ne 0 ]; then
	if [ ${RE_VERBOSE} -ne 0 ]; then
		echo "Compressing..."
	fi
	mkdir -p ${TMPDIR}
	RE_NEW_NAME=`echo ${RE_FILE_NAME} |sed -e 's/\.tif$/.j.tif/'`
	gdal_translate "${RE_FILE_NAME}" "${TMPDIR}/${RE_NEW_NAME}" -co "COMPRESS=JPEG" -co "JPEG_QUALITY=90" -co "TILED=YES" >/dev/null 2>&1
	if [ -f "${TMPDIR}/${RE_NEW_NAME}" ]; then
		RE_FILE_DIR=${TMPDIR}
		RE_FILE_NAME=${RE_NEW_NAME}
	else
		echo "Warning: Failed to compress GeoTIFF"
	fi
fi

# Change to the dir with the file
cd "${RE_FILE_DIR}"
echo "Connecting to ${SERVER}..."

# Check if the server is ready to receive the file
if [ ${RE_VERBOSE} -ne 0 ]; then
	echo "Checking upload availability"
fi
BYTES=`/bin/ls -Lln "${RE_FILE_NAME}" |awk '{print $5}'`
COMMAND="${CURL} -L --insecure -s ${SERVER}/upload/re_upload?bytes=${BYTES}"

if [ ${RE_VERBOSE} -ne 0 ]; then
	echo "Running: ${COMMAND}"
fi
SUCCESS=`${COMMAND} -o - 2>/dev/null |head -1`
if [ "${SUCCESS}" -eq "${SUCCESS}" ] 2>/dev/null; then
	if [ "${SUCCESS}" = "-1" ]; then
		echo "  Server cannot accept the file, it is too large!"
		cleanup
		exit 3
	fi
	if [ "${SUCCESS}" = "2" ]; then
		echo "  Server has already received a file with this signature, use -f to force upload"
		cleanup
		exit 2
	fi
	if [ "${SUCCESS}" = "3" ]; then
		echo "  Server is currently ingesting a file with this name, use -f to force upload"
		cleanup
		exit 3
	fi
	while [ "${SUCCESS}" != "1" ]; do
		if [ ${RE_ONCE} -ne 0 ]; then
			echo "  Server cannot accept the file at this time, try again later"
			exit 3
		fi
		SLEEP=$((${RANDOM} * 5 / 32767 + 10));
		echo "  Server cannot accept the file at this time, trying again in ${SLEEP} seconds..."
		sleep ${SLEEP}
		SUCCESS=`${COMMAND} -o - |head -1`
	done
else
	if [ ${RE_VERBOSE} -ne 0 ]; then
		echo "  Server does not understand file size check, continuing..."
	fi
fi

# Send the file
echo "Sending ${RE_FILE_NAME} (${BYTES} bytes)"
COMMAND="${TIMEOUT} ${CURL} -L --max-time 600 --write-out %{http_code} --silent --fail --insecure ${SERVER}/upload/ -F file=@${RE_FILE_NAME}"
if [ -n "${RE_NAME}" ]; then
	COMMAND="${COMMAND} -F name=${RE_NAME}"
	echo "  Name: ${RE_NAME}"
fi
if [ -n "${RE_DATE}" ]; then
	COMMAND="${COMMAND} -F date=${RE_DATE}"
	echo "  Date: ${RE_DATE}"
fi
if [ -n "${RE_TIME}" ]; then
	COMMAND="${COMMAND} -F time=${RE_TIME}"
	echo "  Time: ${RE_TIME}"
fi
if [ ${RE_PART} -ne 0 ]; then
	COMMAND="${COMMAND} -F part=${RE_PART}"
	if [ ${RE_PART} -gt 0 ]; then
		echo "  Part: ${RE_PART}"
	else
		echo "  Part"
	fi
fi
if [ ${RE_FORCE} -ne 0 ]; then
        COMMAND="${COMMAND} -F force=1"
        if [ ${RE_FORCE} -gt 0 ]; then
        	COMMAND="${COMMAND} -F cast=${RE_FORCE}"
                echo "  Force: ${RE_FORCE}h"
        else
                echo "  Force"
        fi
fi
if [ ${RE_JPEG} -ne 0 ]; then
	echo "  JPEG compressed"
fi
if [ -n "${KEY}" ]; then
	COMMAND="${COMMAND} -F key=${KEY}"
fi
if [ -n "${URI}" ]; then
	B64URI=`echo "${URI}" |${BASE64} |tr -d '\n'`
	COMMAND="${COMMAND} -F uri=\"${B64URI}\" -F uri64=true"
fi

# Retry a few times...
RETRIES=3
if [ ${RE_ONCE} -ne 0 ]; then
	RETRIES=0
fi
if [ ${RE_VERBOSE} -ne 0 ]; then
	echo "Running: ${COMMAND}"
fi
CODE=$(${COMMAND} |head -n 1 |sed -e 's/.*\s//')
LASTEXIT=$?
if [ $((${CODE}+0)) -ge 400 ]; then
	LASTEXIT=${CODE}
fi
while [ ${LASTEXIT} -ne 0 -a ${RETRIES} -gt 0 ]; do
	echo "Curl command failed: ${LASTEXIT}, HTTP code: ${CODE}"
	if [ ${CODE} -ge 400 -a ${CODE} -lt 500 ]; then
		if [ ${CODE} -eq 400 ]; then
			echo "Invalid filename"
			break;
		elif [ ${CODE} -eq 401 ]; then
			echo "Authorization failed"
			break;
		elif [ ${CODE} -eq 409 ]; then
			echo "Conflict"
			break;
		fi
		echo "Client error"
		break;
	fi
	SLEEP=$((${RANDOM} * 30 / 32767))
	echo "Sleeping for ${SLEEP} seconds..."
	sleep ${SLEEP}
	echo "Trying again..."
	CODE=$(${COMMAND} |head -n 1 |sed -e 's/.*\s//')
	LASTEXIT=$?
	if [ $((${CODE}+0)) -ge 400 ]; then
		LASTEXIT=${CODE}
	fi
	RETRIES=$((${RETRIES} - 1))
done
if [ ${RE_VERBOSE} -ne 0 ]; then
	echo "HTTP code: ${CODE}"
	echo "CURL exit: ${LASTEXIT}"
fi
if [ ${LASTEXIT} -eq 0 ]; then
	echo "Done"
else
	echo "Giving up"
fi

cleanup

exit ${LASTEXIT}